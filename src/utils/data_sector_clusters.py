"""
Helper script to load and process finance sector allocations from IATI data.
Creates sector cluster features using embeddings and clustering.

Returns DataFrame with:
- sector_hhi: Herfindahl-Hirschman Index (spending concentration)
- sector_cluster_{cluster_name}: allocation % for each cluster
- sector_cluster_{special_sector}: direct allocation for special non-clustered sectors
- n_sectors: number of sectors per activity
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Special sectors that get their own columns and are NOT embedded/clustered
SPECIAL_SECTORS = [
    "increased food production",
    # 'reduced PM2.5 air pollution',
    # 'more people with access to electricity',
]


def _parse_sector_records(
    path: Path, skip_ids: set, activity_records: list, sector_to_activities: dict
):
    """Read one JSONL sectors file, appending to activity_records and sector_to_activities.

    Skips activity_ids already in skip_ids (so primary file takes precedence over fallback).
    Returns the set of activity_ids added.
    """
    added = set()
    with open(path) as f:
        for line in f:
            data = json.loads(line.strip())
            activity_id = data.get("activity_id")
            if not activity_id or activity_id in skip_ids:
                continue

            response_text = data.get("response_text", "{}")
            try:
                parsed = json.loads(response_text)
            except Exception:
                continue

            allocations = parsed.get("quantitative_outcome_allocations", [])
            if not allocations:
                continue

            sector_amounts = {}
            total_amount = 0
            for alloc in allocations:
                outcome = alloc.get("outcome", "")
                custom = alloc.get("custom_outcome", "")
                amount = alloc.get("amount_allocated", 0)
                sector_label = outcome if (outcome and outcome != "other") else custom
                if not sector_label or sector_label.strip() == "":
                    continue
                sector_amounts[sector_label] = (
                    sector_amounts.get(sector_label, 0) + amount
                )
                total_amount += amount

            if total_amount == 0:
                continue

            sector_props = {k: v / total_amount for k, v in sector_amounts.items()}
            hhi = sum(prop**2 for prop in sector_props.values())

            activity_records.append(
                {
                    "activity_id": activity_id,
                    "sector_hhi": hhi,
                    "n_sectors": len(sector_props),
                    "sector_props": sector_props,
                }
            )
            added.add(activity_id)

            for sector_label, prop in sector_props.items():
                if sector_label not in sector_to_activities:
                    sector_to_activities[sector_label] = []
                sector_to_activities[sector_label].append((activity_id, prop))

    return added


def process_finance_sectors_to_clusters(
    finance_file: Path,
    fallback_file: Path = None,
    embeddings_cache: Path = None,
    n_clusters: int = 10,
    force_recompute: bool = False,
    train_activity_ids=None,
) -> pd.DataFrame:
    """
    Embed sector labels, cluster them, and create activity features.
    Returns DataFrame with activity_id as index: sector_hhi, n_sectors, sector_cluster_* columns.
    """
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    if not finance_file.exists():
        raise FileNotFoundError(f"Finance sectors file not found: {finance_file}")

    activity_records = []  # One per activity
    sector_to_activities = {}  # sector_label -> [(activity_id, proportion), ...]

    primary_ids = _parse_sector_records(
        finance_file,
        skip_ids=set(),
        activity_records=activity_records,
        sector_to_activities=sector_to_activities,
    )
    print(f"  Loaded {len(primary_ids)} activities with sector data from primary file")

    if fallback_file is not None and fallback_file.exists():
        fallback_ids = _parse_sector_records(
            fallback_file,
            skip_ids=primary_ids,
            activity_records=activity_records,
            sector_to_activities=sector_to_activities,
        )
        print(
            f"  Loaded {len(fallback_ids)} additional activities from fallback file (finance text)"
        )
    elif fallback_file is not None:
        raise FileNotFoundError(
            f"Required fallback finance sectors file not found: {fallback_file}\n"
            f"Copy it from forecasting_iati: cp /home/dmrivers/Code/forecasting_iati/data/outputs_finance_sectors_from_finance_text.jsonl {fallback_file}"
        )

    if not activity_records:
        print("No finance sector records found")
        return None

    all_sector_labels = set(sector_to_activities.keys())
    special_sectors_found = [s for s in SPECIAL_SECTORS if s in all_sector_labels]
    regular_sector_labels = sorted(
        [s for s in all_sector_labels if s not in SPECIAL_SECTORS]
    )

    unique_sector_labels = regular_sector_labels
    sector_embeddings = None

    if embeddings_cache and embeddings_cache.exists() and not force_recompute:
        try:
            with open(embeddings_cache, "rb") as f:
                cache_data = pickle.load(f)
                sector_embeddings = cache_data["embeddings"]
                cached_labels = cache_data["sector_labels"]

                # Verify cache matches current data
                if cached_labels != unique_sector_labels:
                    sector_embeddings = None
        except:
            sector_embeddings = None

    if sector_embeddings is None:
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5", embed_batch_size=64
        )

        sector_embeddings = []
        batch_size = 64
        for i in range(0, len(unique_sector_labels), batch_size):
            batch = unique_sector_labels[i : i + batch_size]
            batch_embs = [embed_model.get_text_embedding(label) for label in batch]
            sector_embeddings.extend(batch_embs)

        sector_embeddings = np.array(sector_embeddings)

        if embeddings_cache:
            embeddings_cache.parent.mkdir(parents=True, exist_ok=True)
            with open(embeddings_cache, "wb") as f:
                pickle.dump(
                    {
                        "embeddings": sector_embeddings,
                        "sector_labels": unique_sector_labels,
                    },
                    f,
                )

    # Determine which sector labels to use for KMeans fitting (train-only to avoid leakage)
    if train_activity_ids is not None:
        fit_sector_labels = sorted(
            [
                label
                for label in regular_sector_labels
                if any(
                    act_id in train_activity_ids
                    for act_id, _ in sector_to_activities[label]
                )
            ]
        )
        print(
            f"  KMeans fitting on {len(fit_sector_labels)} train-period sector labels "
            f"(vs {len(regular_sector_labels)} total regular labels)"
        )
        label_to_idx = {label: i for i, label in enumerate(regular_sector_labels)}
        fit_indices = np.array([label_to_idx[label] for label in fit_sector_labels])
        fit_embeddings = sector_embeddings[fit_indices]
    else:
        fit_sector_labels = regular_sector_labels
        fit_embeddings = sector_embeddings

    # Cluster: fit on fit_sector_labels, predict for all regular sector labels
    n_clusters_actual = min(n_clusters, len(fit_sector_labels))
    kmeans = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init=10)
    kmeans.fit(fit_embeddings)
    cluster_labels = kmeans.predict(sector_embeddings)

    # Map sector_label -> cluster_id
    sector_to_cluster = {
        label: int(cluster)
        for label, cluster in zip(regular_sector_labels, cluster_labels, strict=False)
    }

    # Find most common sector_label in each cluster for naming
    cluster_names = {}
    for cluster_id in range(n_clusters_actual):
        sectors_in_cluster = [
            label for label, cid in sector_to_cluster.items() if cid == cluster_id
        ]
        # Count total usage (sum of all activities using each sector)
        sector_usage = {
            sector: len(sector_to_activities[sector]) for sector in sectors_in_cluster
        }
        most_common = (
            max(sector_usage.items(), key=lambda x: x[1])[0]
            if sector_usage
            else f"cluster_{cluster_id}"
        )
        # Clean name for column
        clean_name = (
            most_common.replace(" ", "_")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")[:50]
        )
        cluster_names[cluster_id] = clean_name

    for record in activity_records:
        sector_props = record["sector_props"]

        for special_sector in special_sectors_found:
            clean_name = (
                special_sector.replace(" ", "_")
                .replace("-", "_")
                .replace("(", "")
                .replace(")", "")[:50]
            )
            record[f"sector_cluster_{clean_name}"] = 0.0

        for cluster_id in range(n_clusters_actual):
            cluster_name = cluster_names[cluster_id]
            record[f"sector_cluster_{cluster_name}"] = 0.0

        # Assign allocations to special columns OR clusters
        for sector_label, prop in sector_props.items():
            if sector_label in special_sectors_found:
                clean_name = (
                    sector_label.replace(" ", "_")
                    .replace("-", "_")
                    .replace("(", "")
                    .replace(")", "")[:50]
                )
                record[f"sector_cluster_{clean_name}"] = prop
            else:
                # Sum allocations for regular sector clusters
                cluster_id = sector_to_cluster.get(sector_label)
                if cluster_id is not None:
                    cluster_name = cluster_names[cluster_id]
                    record[f"sector_cluster_{cluster_name}"] += prop

        del record["sector_props"]

    df = pd.DataFrame(activity_records).set_index("activity_id")

    return df
