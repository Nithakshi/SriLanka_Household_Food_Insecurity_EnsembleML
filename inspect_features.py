
from pathlib import Path
import pyreadstat

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FILES = {
    "Identification": "mod_a_household_identification.dta",
    "Housing": "mod_c_housing.dta",
    "Assets": "mod_d_assets.dta",
    "Debt": "mod_n1_debt.dta",
    "Savings": "mod_n2_savings.dta",
}

IMPORTANT_FEATURES = [
    "d_12",
    "d_15",
    "d_06",
    "d_16",
    "d_03",
    "d_14",
    "d_22",
    "d_starttime",
    "hhweight",
    "ppweight",
    "adweight",
    "a_02_district",
    "c1_03",
    "c1_09",
    "c1_02",
    "n2_02_3",
]

for module, filename in FILES.items():
    path = DATA_DIR / filename

    _, metadata = pyreadstat.read_dta(
        str(path),
        metadataonly=True,
    )

    available = set(metadata.column_names)

    for feature in IMPORTANT_FEATURES:
        if feature not in available:
            continue

        print("\n" + "=" * 60)
        print("Module:", module)
        print("Feature:", feature)

        label = metadata.column_names_to_labels.get(
            feature,
            "No description available",
        )

        print("Description:", label)

        label_set = metadata.variable_to_label.get(feature)

        if label_set:
            values = metadata.value_labels.get(
                label_set, {}
            )
            print("Response meanings:", values)

print("\nFeature inspection completed!")
