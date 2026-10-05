import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def extract_cbc_json(extracted_text):

    prompt = f"""
You are a medical laboratory report extraction system.

Extract the CBC values from the OCR text.

IMPORTANT:

1. Extract ONLY values explicitly present in the OCR text.
2. NEVER invent a missing value.
3. NEVER calculate a medical value.
4. If a parameter is present in the OCR text, it MUST be extracted.
5. Pay special attention to:
   - TOTAL RBC COUNT
   - PLATELET COUNT
   - HEMOGLOBIN
   - TOTAL LEUKOCYTE COUNT
6. The report may use different units. Normalize them according to
   the requested output schema.
7. Do not provide a diagnosis.
8. Return ONLY JSON.

UNIT NORMALIZATION:

Hemoglobin:
- Always return g/dL.

RBC:
- If the report says million/cumm, million/mm3, or million/µL,
  return million/µL.
- Example:
  5 million/cumm -> 5 million/µL

WBC:
- Always return /µL.
- Example:
  5,100 cumm -> 5100 /µL

Platelets:
- Always return /µL.
- Example:
  3.5 lakhs/cumm -> 350000 /µL
  3.5 lakh/cumm -> 350000 /µL
  350 x 10^3/µL -> 350000 /µL

Hematocrit:
- Always return %.

MCV:
- Always return fL.

MCH:
- Always return pg.

MCHC:
- Always return %.

Differential counts:
- Neutrophils, lymphocytes, monocytes, eosinophils and basophils
  should always be returned as %.

IMPORTANT:
The words "lakhs", "lakh", "million", "cumm", etc. may appear
in the OCR text. Normalize the numerical value and unit correctly.

OCR TEXT:
-------------------------
{extracted_text}
-------------------------
"""

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "system",
                "content": (
                    "You extract structured laboratory data. "
                    "You must follow the JSON schema exactly."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "cbc_report",
                "strict": True,

                "schema": {

                    "type": "object",

                    "properties": {

                        "patient": {
                            "type": "object",
                            "properties": {

                                "name": {
                                    "type": ["string", "null"]
                                },

                                "age": {
                                    "type": ["number", "null"]
                                },

                                "gender": {
                                    "type": ["string", "null"]
                                }
                            },

                            "required": [
                                "name",
                                "age",
                                "gender"
                            ],

                            "additionalProperties": False
                        },

                        "cbc": {
                            "type": "object",

                            "properties": {

                                "hemoglobin": {
                                    "type": ["object", "null"],
                                    "properties": {
                                        "value": {
                                            "type": ["number", "null"]
                                        },
                                        "unit": {
                                            "type": ["string", "null"]
                                        }
                                    },
                                    "required": [
                                        "value",
                                        "unit"
                                    ],
                                    "additionalProperties": False
                                },

                                "rbc": {
                                    "type": ["object", "null"],
                                    "properties": {
                                        "value": {
                                            "type": ["number", "null"]
                                        },
                                        "unit": {
                                            "type": ["string", "null"]
                                        }
                                    },
                                    "required": [
                                        "value",
                                        "unit"
                                    ],
                                    "additionalProperties": False
                                },

                                "wbc": {
                                    "type": ["object", "null"],
                                    "properties": {
                                        "value": {
                                            "type": ["number", "null"]
                                        },
                                        "unit": {
                                            "type": ["string", "null"]
                                        }
                                    },
                                    "required": [
                                        "value",
                                        "unit"
                                    ],
                                    "additionalProperties": False
                                },

                                "platelets": {
                                    "type": ["object", "null"],
                                    "properties": {
                                        "value": {
                                            "type": ["number", "null"]
                                        },
                                        "unit": {
                                            "type": ["string", "null"]
                                        }
                                    },
                                    "required": [
                                        "value",
                                        "unit"
                                    ],
                                    "additionalProperties": False
                                },

                                "hematocrit": {
                                    "type": ["object", "null"],
                                    "properties": {
                                        "value": {
                                            "type": ["number", "null"]
                                        },
                                        "unit": {
                                            "type": ["string", "null"]
                                        }
                                    },
                                    "required": [
                                        "value",
                                        "unit"
                                    ],
                                    "additionalProperties": False
                                },

                                "mcv": {
                                    "type": ["object", "null"],
                                    "properties": {
                                        "value": {
                                            "type": ["number", "null"]
                                        },
                                        "unit": {
                                            "type": ["string", "null"]
                                        }
                                    },
                                    "required": [
                                        "value",
                                        "unit"
                                    ],
                                    "additionalProperties": False
                                },

                                "mch": {
                                    "type": ["object", "null"],
                                    "properties": {
                                        "value": {
                                            "type": ["number", "null"]
                                        },
                                        "unit": {
                                            "type": ["string", "null"]
                                        }
                                    },
                                    "required": [
                                        "value",
                                        "unit"
                                    ],
                                    "additionalProperties": False
                                },

                                "mchc": {
                                    "type": ["object", "null"],
                                    "properties": {
                                        "value": {
                                            "type": ["number", "null"]
                                        },
                                        "unit": {
                                            "type": ["string", "null"]
                                        }
                                    },
                                    "required": [
                                        "value",
                                        "unit"
                                    ],
                                    "additionalProperties": False
                                },

                                "neutrophils": {
                                    "type": ["object", "null"],
                                    "properties": {
                                        "value": {
                                            "type": ["number", "null"]
                                        },
                                        "unit": {
                                            "type": ["string", "null"]
                                        }
                                    },
                                    "required": [
                                        "value",
                                        "unit"
                                    ],
                                    "additionalProperties": False
                                },

                                "lymphocytes": {
                                    "type": ["object", "null"],
                                    "properties": {
                                        "value": {
                                            "type": ["number", "null"]
                                        },
                                        "unit": {
                                            "type": ["string", "null"]
                                        }
                                    },
                                    "required": [
                                        "value",
                                        "unit"
                                    ],
                                    "additionalProperties": False
                                },

                                "monocytes": {
                                    "type": ["object", "null"],
                                    "properties": {
                                        "value": {
                                            "type": ["number", "null"]
                                        },
                                        "unit": {
                                            "type": ["string", "null"]
                                        }
                                    },
                                    "required": [
                                        "value",
                                        "unit"
                                    ],
                                    "additionalProperties": False
                                },

                                "eosinophils": {
                                    "type": ["object", "null"],
                                    "properties": {
                                        "value": {
                                            "type": ["number", "null"]
                                        },
                                        "unit": {
                                            "type": ["string", "null"]
                                        }
                                    },
                                    "required": [
                                        "value",
                                        "unit"
                                    ],
                                    "additionalProperties": False
                                },

                                "basophils": {
                                    "type": ["object", "null"],
                                    "properties": {
                                        "value": {
                                            "type": ["number", "null"]
                                        },
                                        "unit": {
                                            "type": ["string", "null"]
                                        }
                                    },
                                    "required": [
                                        "value",
                                        "unit"
                                    ],
                                    "additionalProperties": False
                                }
                            },

                            "required": [
                                "hemoglobin",
                                "rbc",
                                "wbc",
                                "platelets",
                                "hematocrit",
                                "mcv",
                                "mch",
                                "mchc",
                                "neutrophils",
                                "lymphocytes",
                                "monocytes",
                                "eosinophils",
                                "basophils"
                            ],

                            "additionalProperties": False
                        }
                    },

                    "required": [
                        "patient",
                        "cbc"
                    ],

                    "additionalProperties": False
                }
            }
        },

        temperature=0
    )

    result = response.choices[0].message.content

    return json.loads(result)