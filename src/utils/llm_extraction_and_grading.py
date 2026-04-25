"""
Drives LLM prompt construction, API calls (OpenAI and Gemini), and response
extraction for generating and grading forecast narratives.
"""

import asyncio
import json
import os
import pprint
import sys
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
from google import genai  # loads the package once

types = genai.types  # just an attribute access, no re-import


import tempfile

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    PdfReader = None
    PdfWriter = None

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from concurrent.futures import ThreadPoolExecutor


def make_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=CONCURRENCY, thread_name_prefix="genai")


# ---- Vertex tuned model support ----
USE_VERTEX = False

PRINT_PROMPT_OPENAI = True
PRINT_PROMPT_BEFORE_UPLOAD = True
OPEN_UPLOADED_WITH_EVINCE = False

AIRPLANE_MODE = False


# Toggle between live calls and Gemini Batch API
BATCH_MODE = False  # False when you want live calls

if BATCH_MODE:
    os.makedirs("../../data/batch_requests", exist_ok=True)

# BATCH_JSONL = "../../data/batch_requests/summaries_batch_requests.jsonl"

CONCURRENCY = 5  # run up to 5 activities at once
# CONCURRENCY = 3  # run up to 5 activities at once

LOCATION_PDFS = "../../data/iati_all_pdfs"
MODEL_NAME = "gemini-2.5-flash"
# MODEL_NAME        = "gemini-3-pro-preview" # can be overwritten
TIMEOUT_SECONDS = 300

if USE_VERTEX:
    VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
    VERTEX_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    # If set, this endpoint is used as the model for Gemini calls.
    # Example: export GEMINI_TUNED_ENDPOINT_ID=3279411077586092032
    GEMINI_TUNED_ENDPOINT_ID = "3279411077586092032"


def get_key(row):
    return row.get("activity_id")
    # return (obj.get("activity_id"), obj.get("cached_file"), obj.get("page_start"))


# ---- Gemini client & structured schemas ----
def make_genai_client() -> genai.Client:
    if USE_VERTEX:
        if not VERTEX_PROJECT:
            raise RuntimeError("Missing GOOGLE_CLOUD_PROJECT for Vertex mode.")
        return genai.Client(
            vertexai=True,
            project=VERTEX_PROJECT,
            location=VERTEX_LOCATION,
            http_options=types.HttpOptions(api_version="v1"),
        )

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get(
        "GOOGLE_API_KEY_GEMINI"
    )
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY (or GOOGLE_API_KEY_GEMINI).")
    return genai.Client(api_key=api_key)


def resolve_gemini_model(model: str) -> str:
    # If caller explicitly passed a Vertex endpoint resource already, keep it.
    if model.startswith("projects/") and "/endpoints/" in model:
        return model

    # If env var says "use tuned endpoint", override base model name.
    if USE_VERTEX:
        if GEMINI_TUNED_ENDPOINT_ID:
            if not VERTEX_PROJECT:
                raise RuntimeError(
                    "GEMINI_TUNED_ENDPOINT_ID set but GOOGLE_CLOUD_PROJECT missing."
                )
            return f"projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}/endpoints/{GEMINI_TUNED_ENDPOINT_ID}"

    # Otherwise fall back to whatever was passed (e.g. 'gemini-2.5-flash')
    return model


# Prompt + response schema builders for Gemini structured output.
# One prompt per CSV row: if both quantitative outcomes and ratings are flagged,
# we combine into a single prompt & a combined schema.
def wait_file_active(client, uploaded, *, timeout=60, interval=1.0):
    """Poll the uploaded file until ACTIVE, or raise on FAILED/timeout."""
    t0 = time.time()
    # Some client versions expose 'name' (e.g., 'files/abc123'), others 'id'
    file_name = getattr(uploaded, "name", None) or uploaded.id
    while True:
        f = client.files.get(name=file_name)  # raises if not found
        state = getattr(f, "state", getattr(f, "display_state", None))
        if state == "ACTIVE":
            return f
        if state == "FAILED":
            raise RuntimeError(
                f"Uploaded file failed to process on server: {file_name}"
            )
        if time.time() - t0 > timeout:
            raise TimeoutError(
                f"Timed out waiting for file to become ACTIVE: {file_name} (last state={state})"
            )
        time.sleep(interval)


async def assemble_and_upload_activity_pdf(bundle: dict[str, Any], client, executor):
    with tempfile.TemporaryDirectory() as tmpdir:
        slice_path_unedited = os.path.join(tmpdir, "combined.pdf")
        try:
            writer = PdfWriter()

            # SORT BY FILE NAME AND PAGE, BEFORE MERGING INTO A DOCUMENT
            items_sorted = sorted(
                bundle["items"], key=lambda x: (x["cached_file"], int(x["page_start"]))
            )

            for item in items_sorted:
                abs_path = os.path.join(LOCATION_PDFS, item["cached_file"])
                page_idx0 = int(item["page_start"]) - 1
                reader = PdfReader(abs_path)
                writer.add_page(reader.pages[page_idx0])

            # for item in bundle["items"]:
            #     abs_path = os.path.join(LOCATION_PDFS, item["cached_file"])
            #     page_idx0 = int(item["page_start"]) - 1
            #     reader = PdfReader(abs_path)
            #     writer.add_page(reader.pages[page_idx0])

            with open(slice_path_unedited, "wb") as fout:
                writer.write(fout)

            slice_path = slice_path_unedited

            if OPEN_UPLOADED_WITH_EVINCE:
                pass
        except Exception as e:
            err = {"ERROR": f"slice_error: {e}"}
            print(err["ERROR"])
            return err

        try:
            loop = asyncio.get_running_loop()

            # wrap to preserve keyword args while using our executor
            def _upload():
                return client.files.upload(file=slice_path)

            def _wait_active(uploaded_file):
                return wait_file_active(
                    client, uploaded_file, timeout=120, interval=0.5
                )

            # upload and poll ACTIVE BEFORE leaving the tempdir
            uploaded = await asyncio.wait_for(
                loop.run_in_executor(executor, _upload),
                timeout=TIMEOUT_SECONDS,
            )
            await asyncio.wait_for(
                loop.run_in_executor(executor, lambda: _wait_active(uploaded)),
                timeout=TIMEOUT_SECONDS + 10,
            )
        except asyncio.TimeoutError:
            return {"ERROR": "upload_timeout"}
        except Exception as e:
            err = {"ERROR": f"upload_error: {e}"}
            print(err["ERROR"])
            return err

    print("uploaded!")
    return uploaded


async def run_one_row(
    response_schema, prompt, row, client, seen_keys, output_jsonl, execpool, model
):
    obj: dict[str, Any] = {}

    # Skip if this row was already processed before
    key = get_key(row)
    if key in seen_keys:
        # print(f"key was seen: {key}. Skipping.")
        return None
    seen_keys.add(key)

    if "num_pages" in row.keys():
        # Build base obj
        obj.update(
            {
                "activity_id": row["activity_id"],
                "section": row.get("section"),
                "num_pages": len(row.get("items", [])),
            }
        )
    else:
        obj.update(
            {
                "activity_id": row["activity_id"],
            }
        )

    system_instruction = None
    text_prompt = prompt
    prompt_type = ""
    if isinstance(prompt, dict):
        system_instruction = prompt.get("system_msg")
        text_prompt = prompt.get("prompt")
        prompt_type = prompt.get("prompt_type", "")

    if PRINT_PROMPT_BEFORE_UPLOAD:
        pprint.pprint(f"prompt for activity id {row['activity_id']}")
        pprint.pprint(prompt)

    uploaded = None
    if row.get("items"):
        # Upload / assemble
        uploaded = await assemble_and_upload_activity_pdf(row, client, execpool)
        if isinstance(uploaded, dict) and "ERROR" in uploaded:
            # propagate the uploader's error shape
            return {**obj, **uploaded}

    activity_upload = None
    evaluation_upload = None

    # Upload activity pages
    if row.get("activity_items"):
        bundle = {"items": row["activity_items"]}
        activity_upload = await assemble_and_upload_activity_pdf(
            bundle, client, execpool
        )
        if isinstance(activity_upload, dict) and "ERROR" in activity_upload:
            return {**obj, **activity_upload}

    # Upload evaluation pages
    if row.get("evaluation_items"):
        bundle = {"items": row["evaluation_items"]}
        evaluation_upload = await assemble_and_upload_activity_pdf(
            bundle, client, execpool
        )
        if isinstance(evaluation_upload, dict) and "ERROR" in evaluation_upload:
            return {**obj, **evaluation_upload}

    # ---- Batch mode branch (no live model call) ----
    if BATCH_MODE:
        # Build "parts" exactly like in the ranker: text prompt + fileData blocks
        parts = []

        if system_instruction is not None:
            parts.append({"text": f"SYSTEM INSTRUCTION:\n{system_instruction}"})

        parts.append({"text": text_prompt})

        # Attach combined activity / evaluation / generic bundles
        if activity_upload:
            parts.append({"text": "ACTIVITY DOCUMENTS:"})
            parts.append(
                {
                    "fileData": {
                        "fileUri": activity_upload.uri,
                        "mimeType": activity_upload.mime_type,
                    }
                }
            )

        if evaluation_upload:
            parts.append({"text": "EVALUATION DOCUMENTS:"})
            parts.append(
                {
                    "fileData": {
                        "fileUri": evaluation_upload.uri,
                        "mimeType": evaluation_upload.mime_type,
                    }
                }
            )

        if uploaded and not activity_upload and not evaluation_upload:
            parts.append({"text": "ACTIVITY DOCUMENTS:"})
            parts.append(
                {
                    "fileData": {
                        "fileUri": uploaded.uri,
                        "mimeType": uploaded.mime_type,
                    }
                }
            )

        request_obj: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
        }

        if response_schema is not None:
            request_obj["generationConfig"] = {
                "responseMimeType": "application/json",
                "responseJsonSchema": response_schema,
            }

        batch_key = f"{obj['activity_id']}::{obj.get('section') or 'NA'}"

        line = {
            "key": batch_key,
            "request": request_obj,
        }

        out_path = Path(output_jsonl)
        batch_dir = Path("../../data/batch_requests")
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_path = batch_dir / f"{out_path.stem}_batch{out_path.suffix}"

        with open(batch_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line))
            f.write("\n")

        print(
            f"Wrote batch request for {obj['activity_id']} / {obj.get('section')} with key={batch_key}"
        )
        return None

    # model call via our execpool
    try:
        loop = asyncio.get_running_loop()

        def _sync_call():
            # ---- Vertex path (tuned endpoint etc.) ----
            if USE_VERTEX:

                def _sync_call():
                    if USE_VERTEX:
                        cfg = types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.2,
                            response_mime_type=(
                                "application/json"
                                if response_schema is not None
                                else None
                            ),
                            response_schema=(
                                response_schema if response_schema is not None else None
                            ),
                        )
                        return client.models.generate_content(
                            model=model,
                            contents=[
                                types.Content(
                                    role="user", parts=[types.Part(text=text_prompt)]
                                )
                            ],
                            config=cfg,
                        )

            # contents = [text_prompt, uploaded] if uploaded else [text_prompt]
            contents = [text_prompt]

            # Add labeled activity documents
            if activity_upload:
                contents.append("ACTIVITY DOCUMENTS:")
                contents.append(
                    {
                        "fileData": {
                            "fileUri": activity_upload.uri,
                            "mimeType": activity_upload.mime_type,
                        }
                    }
                )

            # Add labeled evaluation documents
            if evaluation_upload:
                contents.append("EVALUATION DOCUMENTS:")
                contents.append(
                    {
                        "fileData": {
                            "fileUri": evaluation_upload.uri,
                            "mimeType": evaluation_upload.mime_type,
                        }
                    }
                )

            # Fallback: for generic bundles that only have "items" (your KNN script),
            # attach that single combined PDF as well, but only if we didn't already
            # add activity/evaluation docs.
            if uploaded and not activity_upload and not evaluation_upload:
                contents.append("ACTIVITY DOCUMENTS:")
                contents.append(
                    {
                        "fileData": {
                            "fileUri": uploaded.uri,
                            "mimeType": uploaded.mime_type,
                        }
                    }
                )
            if system_instruction is not None:
                gen_config = {
                    "system_instruction": system_instruction,
                }
            else:
                gen_config = {}

            if response_schema is not None:
                # Only enable JSON mode + structured output when you actually pass a schema
                gen_config["response_mime_type"] = "application/json"
                gen_config["response_schema"] = response_schema

            return client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config,
            )

        fut = loop.run_in_executor(execpool, _sync_call)
        response = await asyncio.wait_for(fut, timeout=TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        obj["ERROR"] = "asyncio.exceptions.TimeoutError"
        return obj
    except Exception as e:
        obj["ERROR"] = f"{type(e).__name__}: {e}"
        return obj

    obj["prompt_type"] = prompt_type

    # ---- Token usage + metadata ----
    usage = getattr(response, "usage_metadata", None)
    if usage:
        obj["token_usage"] = {
            "prompt_token_count": getattr(usage, "prompt_token_count", None),
            "thoughts_token_count": getattr(usage, "thoughts_token_count", None),
            "candidates_token_count": getattr(usage, "candidates_token_count", None),
            "total_token_count": getattr(usage, "total_token_count", None),
            # optional breakdown if your SDK provides it
            "prompt_tokens_details": [
                {
                    "modality": getattr(d, "modality", None),
                    "token_count": getattr(d, "token_count", None),
                }
                for d in (getattr(usage, "prompt_tokens_details", []) or [])
            ],
        }

    # optional: keep a couple of handy IDs alongside usage
    obj["model_version"] = getattr(response, "model_version", None)

    pprint.pprint(f"\nResponse from model {model}")
    pprint.pprint(response)
    # ---- Got a response; store serializable bits and validate JSON ----
    text = getattr(response, "text", str(response))
    obj["response_text"] = text

    return obj


async def run_one_row_openai(
    prompt, row, client, seen_keys, output_jsonl, execpool, model
):
    if type(prompt) == dict:
        system_msg = prompt["system_msg"]
        prompt = prompt["prompt"]
    else:
        system_msg = (
            "You are a careful, experienced international aid decision maker.\n"
            "Only reply with an integer between 0 and 100 (no extra text)"
        )

    if model == "gemini":
        model = MODEL_NAME

    obj: dict[str, Any] = {}

    # Skip if this row was already processed before
    key = get_key(row)
    if key in seen_keys:
        print(f"key was seen: {key}. Skipping.")
        return None
    seen_keys.add(key)

    # Build base obj
    obj.update(
        {
            "activity_id": row["activity_id"],
            "section": row.get("section"),
        }
    )
    print("about to call")
    # model call via our execpool
    try:
        loop = asyncio.get_running_loop()

        def _sync_call():
            print("calling..")
            if PRINT_PROMPT_OPENAI:
                pprint.pprint("system_msg")
                pprint.pprint(system_msg)
                pprint.pprint("prompt")
                pprint.pprint(prompt)
            extra = {}
            if model == "deepseek-reasoner":
                extra = {"thinking": {"type": "enabled"}}

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                extra_body=extra,
            )
            if PRINT_PROMPT_OPENAI:
                pprint.pprint("resp")
                pprint.pprint(resp)
            return resp  # .choices[0].message.content.strip()

        fut = loop.run_in_executor(execpool, _sync_call)
        response = await asyncio.wait_for(fut, timeout=TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        obj["ERROR"] = "asyncio.exceptions.TimeoutError"
        return obj
    except Exception as e:
        obj["ERROR"] = f"{type(e).__name__}: {e}"
        return obj

    # --- serialize only what you want ---
    c0 = response.choices[0]
    msg = c0.message

    obj["response"] = {
        "id": response.id,  # drop this if you don't want it
        "content": msg.content,
        "role": getattr(msg, "role", "assistant"),
        "finish_reason": c0.finish_reason,
        "usage": {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
            "total_tokens": getattr(response.usage, "total_tokens", None),
        },
    }
    # remove Nones to keep it tidy
    obj["response"]["usage"] = {
        k: v for k, v in obj["response"]["usage"].items() if v is not None
    }

    return obj


# was: -> Set[Tuple[...]]
def load_seen_keys(output_jsonl_path: str) -> set[str]:
    seen: set[str] = set()
    p = Path(output_jsonl_path)
    if not p.exists():
        raise FileNotFoundError(f"Seen keys JSONL not found: {p}")
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            aid = obj.get("activity_id")
            if not aid:
                continue
            # only consider success as "seen"
            if obj.get("ERROR"):
                continue
            seen.add(str(aid))
    return seen


async def loop_over_rows_to_call_model(
    output_jsonl, rows, prompts, response_schema=None, execpool=None, model="gemini"
):
    is_gemini = (
        model == "gemini" or "gemini" in model.lower() or model.startswith("projects/")
    )
    is_deepseek = model.startswith("deepseek-")

    if not execpool:
        execpool = make_executor()

    output_jsonl_path = Path(output_jsonl)
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    client = None
    if not AIRPLANE_MODE:
        if is_gemini:
            client = make_genai_client()
            base = MODEL_NAME if model == "gemini" else model
            model = resolve_gemini_model(base)

        elif is_deepseek:
            client = OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
            )

        elif model == "chatgpt" or "gpt" in model.lower():
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        else:
            print("error: can only call gemini, deepseek-*, or chatgpt/gpt")
            sys.exit(1)

    # if model == "gemini" or "gemini" in model.lower() or model.startswith("projects/"):
    #     client = make_genai_client()
    #     # normalize "gemini" -> default model, and optionally override to tuned endpoint
    #     base = MODEL_NAME if model == "gemini" else model
    #     model = resolve_gemini_model(base)

    # elif model == "chatgpt" or "gpt" in model.lower():
    #     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # else:
    #     print("error: can only call gemini or chatgpt")
    #     quit()

    # ... rest of your function unchanged ...

    # we may want transaction bounds for later.
    # txn_bounds = load_txn_bounds()
    # act_bounds = load_activity_bounds_from_subset()

    MAX_ATTEMPTS = 2
    CONSECUTIVE_TIMEOUT_LIMIT = 3
    current_execpool = execpool

    # Build planned set once -- rows and prompts don't change between attempts
    planned = set()
    for r in rows:
        if get_key(r) in prompts.keys():
            planned.add(get_key(r))

    for attempt in range(MAX_ATTEMPTS):
        if attempt > 0:
            current_execpool = make_executor()  # fresh pool; old threads drain safely
            print(f"[RETRY] Attempt {attempt + 1} -- retrying errored rows...")

        seen_keys = load_seen_keys(output_jsonl)
        # Run rows concurrently with a semaphore; write outputs line-by-line without locks
        sem = asyncio.Semaphore(CONCURRENCY)
        consecutive_timeouts = [0]
        circuit_open = asyncio.Event()

        tasks = []  # defined before _guard so cancellation can reach all tasks

        # Default-parameter binding captures per-attempt state at definition time
        async def _guard(
            prompt_and_row: list,
            _seen=seen_keys,
            _sem=sem,
            _ct=consecutive_timeouts,
            _co=circuit_open,
            _ep=current_execpool,
            _tasks=tasks,
        ):
            if _co.is_set():
                return
            async with _sem:
                if _co.is_set():
                    return
                try:
                    if AIRPLANE_MODE:
                        from dummy_response_text_generator import (
                            get_dummy_response_text,
                        )

                        activity_id = prompt_and_row[1].get("activity_id", "unknown")
                        dummy_text = get_dummy_response_text(
                            response_schema, prompt_and_row[0], activity_id
                        )
                        dummy_obj = {
                            "activity_id": activity_id,
                            "response_text": dummy_text,
                        }
                        with open(
                            output_jsonl_path, "a", encoding="utf-8", buffering=1
                        ) as f_out:
                            f_out.write(
                                json.dumps(dummy_obj, ensure_ascii=False) + "\n"
                            )
                            f_out.flush()
                        return

                    if is_gemini:
                        obj = await asyncio.wait_for(
                            run_one_row(
                                response_schema,
                                prompt_and_row[0],
                                prompt_and_row[1],
                                client,
                                _seen,
                                output_jsonl,
                                _ep,
                                model,
                            ),
                            timeout=TIMEOUT_SECONDS + 30,
                        )
                    else:
                        obj = await asyncio.wait_for(
                            run_one_row_openai(
                                prompt_and_row[0],
                                prompt_and_row[1],
                                client,
                                _seen,
                                output_jsonl,
                                _ep,
                                model,
                            ),
                            timeout=TIMEOUT_SECONDS + 30,
                        )
                except asyncio.TimeoutError:
                    obj = {
                        "ERROR": "Overall operation timeout",
                        "activity_id": prompt_and_row[1].get("activity_id"),
                    }
                if not obj:
                    return
                is_timeout_err = "timeout" in str(obj.get("ERROR", "")).lower()
                if obj.get("ERROR"):
                    print(f"[ERROR] {obj.get('activity_id')} {obj['ERROR']}")
                    if is_timeout_err:
                        _ct[0] += 1
                        print(
                            f"[CIRCUIT BREAKER] consecutive timeouts: {_ct[0]}/{CONSECUTIVE_TIMEOUT_LIMIT}"
                        )
                        if _ct[0] >= CONSECUTIVE_TIMEOUT_LIMIT:
                            print(
                                f"[CIRCUIT BREAKER] threshold reached -- cancelling {sum(1 for t in _tasks if not t.done())} remaining tasks"
                            )
                            _co.set()
                            for t in _tasks:
                                if not t.done():
                                    t.cancel()
                    else:
                        _ct[0] = 0
                else:
                    _ct[0] = 0

                with open(
                    output_jsonl_path, "a", encoding="utf-8", buffering=1
                ) as f_out:
                    f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    f_out.flush()  # ensure it hits disk even when stdout is redirected

        for r in rows:
            if get_key(r) in prompts.keys():
                prompt_and_row = (prompts[get_key(r)], r)
                tasks.append(asyncio.create_task(_guard(prompt_and_row)))
            else:
                print(f"[DEBUG] no prompt for activity_id={get_key(r)} (skipping row)")

        done = load_seen_keys(output_jsonl)  # only successes, by your definition
        missing_done = sorted(planned - done)
        print(
            f"[DEBUG] attempt={attempt + 1} planned={len(planned)} success_done={len(done)} missing_success={len(missing_done)}"
        )
        Path("planned_but_not_success.txt").write_text(
            "\n".join(missing_done) + "\n", encoding="utf-8"
        )
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception) and not isinstance(
                    r, asyncio.CancelledError
                ):
                    print("Task error:", r)
        print(
            f"[attempt {attempt + 1}] Processed {len(tasks)} rows, output: {output_jsonl_path}."
        )

        if attempt > 0:
            current_execpool.shutdown(
                wait=False
            )  # clean up retry executor; caller owns attempt-0 pool

        done_after = load_seen_keys(output_jsonl)
        still_missing = planned - done_after
        if not still_missing:
            print("[RETRY] All planned rows succeeded.")
            break
        if attempt < MAX_ATTEMPTS - 1:
            print(
                f"[RETRY] {len(still_missing)} rows still missing -- retrying in 10s..."
            )
            await asyncio.sleep(10)  # let network stabilize before retry


####################  fallback for if we can't find some of the ids from our first attempt, just get some early highly ranked pages instead ##########


# Only exclude these (exactly as requested)
EXCLUDED_CAT_PAGES = {
    "glossary",
    "blank_page",
    "table_of_contents",
    "references",
}


from collections import Counter

DEBUG_AID = "44000-P157571"
debug_miss_reasons = Counter()
