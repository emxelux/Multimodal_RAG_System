from llama_index.core.node_parser import SentenceSplitter


def chunk_nodes(parent_nodes):
    child_parser = SentenceSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    child_nodes = child_parser.get_nodes_from_documents(parent_nodes)
        
    for child in child_nodes:
        child.metadata["parent_id"] = child.ref_doc_id
    
    return child_nodes

