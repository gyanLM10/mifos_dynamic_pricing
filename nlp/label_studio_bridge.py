"""
Label Studio Bridge — Utilities for importing/exporting annotation data
between the Mifos X CRM and Label Studio for domain-expert labeling of
client communications.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Label Studio task format for text classification
LABEL_STUDIO_TEMPLATE = {
    "label_config": """
    <View>
      <Header value="Client Communication Risk Assessment"/>
      <Text name="text" value="$text"/>
      <Choices name="risk_level" toName="text" choice="single">
        <Choice value="low_risk"/>
        <Choice value="moderate_risk"/>
        <Choice value="high_risk"/>
        <Choice value="critical_risk"/>
      </Choices>
      <Choices name="risk_factors" toName="text" choice="multiple">
        <Choice value="crop_failure"/>
        <Choice value="medical_emergency"/>
        <Choice value="natural_disaster"/>
        <Choice value="job_loss"/>
        <Choice value="family_crisis"/>
        <Choice value="debt_spiral"/>
        <Choice value="business_failure"/>
        <Choice value="seasonal_hardship"/>
      </Choices>
      <TextArea name="notes" toName="text"
                placeholder="Additional observations..."
                maxSubmissions="1"/>
    </View>
    """,
}


def mifos_notes_to_label_studio_tasks(
    notes: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Convert Mifos X client notes/SMS logs into Label Studio task format.

    Parameters
    ----------
    notes : list[dict]
        List of dicts with keys: 'client_id', 'text', 'date', 'source'
    output_path : str or Path, optional
        If provided, writes the tasks to a JSON file.

    Returns
    -------
    list[dict]
        Label Studio tasks ready for import.
    """
    tasks = []
    for note in notes:
        task = {
            "data": {
                "text": note.get("text", ""),
                "client_id": note.get("client_id", "unknown"),
                "date": note.get("date", ""),
                "source": note.get("source", "crm_note"),
            }
        }
        tasks.append(task)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(tasks, f, indent=2)
        logger.info(f"Exported {len(tasks)} tasks to {output_path}")

    return tasks


def parse_label_studio_annotations(
    annotations_path: str | Path,
) -> list[dict[str, Any]]:
    """
    Parse completed Label Studio annotations into structured training data
    for the sentiment mapper.

    Returns
    -------
    list[dict]
        Each dict has: 'text', 'client_id', 'risk_level', 'risk_factors', 'notes'
    """
    annotations_path = Path(annotations_path)
    with open(annotations_path) as f:
        raw = json.load(f)

    parsed = []
    for item in raw:
        data = item.get("data", {})
        annotations = item.get("annotations", [{}])

        # Take the first (or most recent) annotation
        ann = annotations[0] if annotations else {}
        results = ann.get("result", [])

        risk_level = None
        risk_factors = []
        notes = ""

        for result in results:
            from_name = result.get("from_name", "")
            value = result.get("value", {})

            if from_name == "risk_level":
                choices = value.get("choices", [])
                risk_level = choices[0] if choices else None

            elif from_name == "risk_factors":
                risk_factors = value.get("choices", [])

            elif from_name == "notes":
                text_vals = value.get("text", [])
                notes = text_vals[0] if text_vals else ""

        parsed.append({
            "text": data.get("text", ""),
            "client_id": data.get("client_id", ""),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "notes": notes,
        })

    logger.info(f"Parsed {len(parsed)} annotations from {annotations_path}")
    return parsed


def generate_sample_annotations(
    output_path: str | Path, n_samples: int = 50, seed: int = 42
) -> None:
    """Generate sample annotation data for development."""
    import numpy as np

    rng = np.random.default_rng(seed)

    sample_texts = [
        "My crops failed this season, I don't know how I will repay.",
        "Business is growing well, expecting to pay early.",
        "Had a medical emergency, need an extension on my loan.",
        "Received a promotion at work, income has increased.",
        "The floods destroyed my shop inventory.",
        "Making regular savings now, loan payments on track.",
        "Lost my job last month, struggling to find work.",
        "Good harvest this year, will pay on time.",
        "Family illness has drained our savings.",
        "New business venture is profitable, ready to expand.",
    ]

    risk_levels = ["low_risk", "moderate_risk", "high_risk", "critical_risk"]
    risk_factors_pool = [
        "crop_failure", "medical_emergency", "natural_disaster",
        "job_loss", "family_crisis", "debt_spiral",
        "business_failure", "seasonal_hardship",
    ]

    annotations = []
    for i in range(n_samples):
        text = rng.choice(sample_texts)
        level = rng.choice(risk_levels)
        n_factors = rng.integers(0, 4)
        factors = list(rng.choice(risk_factors_pool, size=n_factors, replace=False))

        annotations.append({
            "data": {
                "text": text,
                "client_id": f"CLI-{i:04d}",
                "date": f"2024-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}",
                "source": "crm_note",
            },
            "annotations": [{
                "result": [
                    {
                        "from_name": "risk_level",
                        "to_name": "text",
                        "type": "choices",
                        "value": {"choices": [level]},
                    },
                    {
                        "from_name": "risk_factors",
                        "to_name": "text",
                        "type": "choices",
                        "value": {"choices": factors},
                    },
                ],
            }],
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=2)
    logger.info(f"Generated {n_samples} sample annotations → {output_path}")
