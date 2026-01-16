#!/usr/bin/env python3
"""Module for searching RCSB PDB using the Search API v2."""

import requests
import logging
from .config import PipelineConfig

logger = logging.getLogger("pipeline.search")

def search_pdb(config: PipelineConfig) -> list[str]:
    """
    Search RCSB PDB for structures matching criteria in the config.

    Args:
        config: The pipeline configuration object.

    Returns:
        A list of PDB IDs.
    """
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    params = config.search_parameters

    query_nodes = []

    # 1. Metal Ion
    if params.metal_ion:
        query_nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_nonpolymer_entity_instance_container_identifiers.comp_id",
                "operator": "exact_match",
                "value": params.metal_ion.upper()
            }
        })

    # 2. Resolution
    if params.resolution_cutoff:
        query_nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal",
                "value": params.resolution_cutoff
            }
        })

    # 3. Experimental Method
    if params.experimental_method and params.experimental_method.upper() != "ALL":
        query_nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "exact_match",
                "value": params.experimental_method.upper()
            }
        })

    # 4. Polymer Type
    if params.polymer_type is not None and str(params.polymer_type).strip().upper() != "ALL":
        pt_raw = str(params.polymer_type).strip()
        pt_upper = pt_raw.upper()

        # RCSB has two common representations:
        # - entity_poly.rcsb_entity_polymer_type: coarse buckets like "Protein", "DNA", "RNA"
        # - entity_poly.type: PDBx/mmCIF polymer types like "polypeptide(L)"
        # Users often provide the latter; handle both.
        coarse = {"PROTEIN", "DNA", "RNA", "NA-HYBRID", "OTHER"}
        if "(" in pt_raw or ")" in pt_raw or pt_upper not in coarse:
            # Assume a PDBx/mmCIF style polymer type (case-sensitive in practice)
            attr = "entity_poly.type"
            value = pt_raw
        else:
            attr = "entity_poly.rcsb_entity_polymer_type"
            # Preserve canonical casing for known values
            if pt_upper == "PROTEIN":
                value = "Protein"
            elif pt_upper == "NA-HYBRID":
                value = "NA-hybrid"
            else:
                value = pt_upper

        query_nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": attr,
                "operator": "exact_match",
                "value": value,
            },
        })

    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": query_nodes
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True}
    }

    logger.info(f"Executing search with parameters: {params.model_dump()}")
    response = requests.post(url, json=query)

    if response.status_code == 200:
        results = response.json()
        pdb_ids = [entry["identifier"] for entry in results.get("result_set", [])]
        logger.info(f"Search returned {len(pdb_ids)} PDB IDs.")
        return pdb_ids
    else:
        logger.error(f"RCSB search failed with status {response.status_code}: {response.text}")
        return []
