def process_document(file_path: Path):
    global active_document

    file_path = file_path.resolve()  # 🔥 IMPORTANT FIX
    filename = file_path.name

   
    docs = load_document(str(file_path))

    if not docs:
        raise RuntimeError("No text extracted from PDF (check file or OCR requirement)")

   
    parent_nodes = create_nodes(docs)

    if not parent_nodes:
        raise RuntimeError("No parent nodes created from document")

    for node in parent_nodes:
        parent_id = str(uuid.uuid4())
        node.metadata["parent_id"] = parent_id

        db.add_document(
            source=filename,
            parent_id=parent_id,
            parent_metadata=node.metadata,
            parent_content=node.get_content(),
        )


    child_nodes = chunk_nodes(parent_nodes)

    if not child_nodes:
        raise RuntimeError("No text chunks generated from document")

    # propagate parent_id safely
    for child in child_nodes:
        if "parent_id" not in child.metadata:
            child.metadata["parent_id"] = (
                child.metadata.get("parent_id")
                or getattr(child, "extra_info", {}).get("parent_id")
            )

    vec_db.upsert_document(child_nodes)

    active_document = filename

    print(f"[INGESTION COMPLETE] {filename}")