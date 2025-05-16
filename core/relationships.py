
from random import shuffle

def create_relationships(tx, df):
    case_data = []
    answer_data, demo_data, submitter_data = [], [], []

    for _, row in df.iterrows():
        case_id = int(row["Case_No"])
        upload_id = str(case_id)
        case_data.append({"id": case_id, "upload_id": upload_id})

        for q in [f"A{i}" for i in range(1, 11)]:
            answer_data.append({"upload_id": upload_id, "q": q, "val": int(row[q])})

        for col in ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD"]:
            demo_data.append({"upload_id": upload_id, "type": col, "val": row[col]})

        submitter_data.append({"upload_id": upload_id, "val": row["Who_completed_the_test"]})

    tx.run("UNWIND $data as row MERGE (c:Case {id: row.id}) SET c.upload_id = row.upload_id, c.embedding = null", data=case_data)
    tx.run("""
        UNWIND $data as row
        MATCH (q:BehaviorQuestion {name: row.q})
        MATCH (c:Case {upload_id: row.upload_id})
        MERGE (c)-[:HAS_ANSWER {value: row.val}]->(q)
    """, data=answer_data)
    tx.run("""
        UNWIND $data as row
        MATCH (d:DemographicAttribute {type: row.type, value: row.val})
        MATCH (c:Case {upload_id: row.upload_id})
        MERGE (c)-[:HAS_DEMOGRAPHIC]->(d)
    """, data=demo_data)
    tx.run("""
        UNWIND $data as row
        MATCH (s:SubmitterType {type: row.val})
        MATCH (c:Case {upload_id: row.upload_id})
        MERGE (c)-[:SUBMITTED_BY]->(s)
    """, data=submitter_data)

def create_similarity_relationships(tx, df, max_pairs=10000):
    pairs = set()
    for i, row1 in df.iterrows():
        for j, row2 in df.iloc[i+1:].iterrows():
            if sum(row1[f'A{k}'] == row2[f'A{k}'] for k in range(1,11)) >= 7:
                pairs.add((int(row1['Case_No']), int(row2['Case_No'])))

    demo_cols = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD']
    for col in demo_cols:
        grouped = df.groupby(col)['Case_No'].apply(list)
        for case_list in grouped:
            for i in range(len(case_list)):
                for j in range(i+1, len(case_list)):
                    pairs.add((int(case_list[i]), int(case_list[j])))

    pair_list = list(pairs)[:max_pairs]
    shuffle(pair_list)

    tx.run("""
        UNWIND $batch AS pair
        MATCH (c1:Case {id: pair.id1}), (c2:Case {id: pair.id2})
        MERGE (c1)-[:SIMILAR_TO]->(c2)
    """, batch=[{'id1':x, 'id2':y} for x,y in pair_list])
