import json
from pathlib import Path

import torch
from transformers import AutoTokenizer



class WikiScriptDataset:
    """
    Dataset of Script-level Counterfactuals.
    Specifically selected from the QA validation slice from Mitchell et al.
    Project page: http://nlp.cs.washington.edu/zeroshot/
    """

    def __init__(self, data_dir: str, tok: AutoTokenizer, size=None, *args, **kwargs):
        data_dir = Path(data_dir)
        script_loc = data_dir / "mquake-t-final.json"
        assert script_loc.exists(), f"{script_loc} does not exist. Check it."

        with open(script_loc, "r") as f:
            raw = json.load(f)

        data = []
        for i, record in enumerate(raw):
            # assert (
            #     "nq question: " in record["loc"]
            # ), f"Neighborhood prompt missing `nq question:`. Check for errors?"
            # ans_toks = tok(" " + record["loc_ans"])["input_ids"]
            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": record["prompt"],
                        "subject": record["subject"],
                        "target_new": {"str": record["new_update"]},
                        "target_true": {"str": record["old_update"]},
                    },
                    "paraphrase_prompts": record["sliced_question"],
                    "neighborhood_prompts": record['neighborhood_prompts'] if 'neighborhood_prompts' in record else [],
                    "attribute_prompts": [],
                    "generation_prompts": record['generation_prompts'] if 'generation_prompts' in record else [],
                }
            )

        self._data = data[:size]

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
