
def create_nodes(tx, df):
    for q in [f"A{i}" for i in range(1, 11)]:
        tx.run("MERGE (:BehaviorQuestion {name: $q})", q=q)

    for column in ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]:
        for val in df[column].dropna().unique():
            tx.run("MERGE (:DemographicAttribute {type: $type, value: $val})", type=column, val=val)

    for val in df["Who_completed_the_test"].dropna().unique():
        tx.run("MERGE (:SubmitterType {type: $val})", val=val)
