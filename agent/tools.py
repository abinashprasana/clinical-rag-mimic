"""Structured, non-LLM-generated tools the agent can call.

fda_label_tool queries the openFDA Drug Label API (free, no API key) for the
FDA-approved label text of a drug. Results are returned as-is from the label
-- never paraphrased by an LLM by default -- so dosage/contraindication
figures shown to a clinician are always traceable to the actual FDA source,
not model-generated text.
"""
import requests
import config

OPENFDA_LABEL_URL = 'https://api.fda.gov/drug/label.json'


def _first(field_list):
    return field_list[0] if field_list else None


def _query_openfda(field, drug_name, exact=False):
    field_query = f'openfda.{field}.exact' if exact else f'openfda.{field}'
    params = {
        'search': f'{field_query}:"{drug_name.upper() if exact else drug_name}"',
        'limit': 1,
    }
    try:
        resp = requests.get(OPENFDA_LABEL_URL, params=params, timeout=config.OPENFDA_TIMEOUT_SECONDS)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    results = resp.json().get('results') or []
    return results[0] if results else None


def fda_label_tool(drug_name):
    """Look up dosage/contraindication/warning text for a drug from the FDA
    label database. Tries an exact field match first (brand, then generic,
    then substance name) to avoid matching an unrelated combination product
    that merely contains the searched name as a substring (e.g. searching
    "lisinopril" should not silently return a lisinopril+hydrochlorothiazide
    combo label) -- falls back to a looser phrase match only if no exact
    match exists anywhere.

    Returns a dict with the FDA label fields, or None if no match was found
    -- callers must treat None as "not found", never fabricate an answer in
    its place.
    """
    drug_name = drug_name.strip()
    if not drug_name:
        return None

    fields = ('brand_name', 'generic_name', 'substance_name')
    record = None
    for field in fields:
        record = _query_openfda(field, drug_name, exact=True)
        if record:
            break
    if not record:
        for field in fields:
            record = _query_openfda(field, drug_name, exact=False)
            if record:
                break

    if not record:
        return None

    openfda = record.get('openfda', {})
    return {
        'drug_name': drug_name,
        'matched_brand_name': _first(openfda.get('brand_name', [])),
        'matched_generic_name': _first(openfda.get('generic_name', [])),
        'dosage_and_administration': _first(record.get('dosage_and_administration', [])),
        'contraindications': _first(record.get('contraindications', [])),
        'warnings': _first(record.get('warnings') or record.get('warnings_and_cautions', [])),
        'source': 'openFDA Drug Label API',
        'label_url': f'https://labels.fda.gov/?search={drug_name}',
    }
