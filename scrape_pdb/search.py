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
    if params.polymer_type and params.polymer_type.upper() != "ALL":
        query_nodes.append({
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "entity_poly.rcsb_entity_polymer_type",
                "operator": "exact_match",
                "value": params.polymer_type
            }
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
