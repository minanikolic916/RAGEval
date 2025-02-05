from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

def load_data(folder_path):
    docs = SimpleDirectoryReader(folder_path).load_data()
    text_parser = SentenceSplitter(
        chunk_size=256,
        chunk_overlap=20
    )
    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(docs):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc = docs[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)
    return nodes

def data_info(folder_path:str):
    nodes = load_data(folder_path)
    print(f"Ukupan broj cvorova: {len(nodes)}\n")
    nodes_doc = {}
    for node in nodes:
        doc_file_name = node.metadata.get("file_name")
        if doc_file_name not in nodes_doc:
            nodes_doc[doc_file_name] = 0
        nodes_doc[doc_file_name] +=1
    for doc_file_name, count in nodes_doc.items():
        print(f"Document *{doc_file_name}* has {count} node/nodes.")

def get_doc_nodes(folder_path:str, doc_file_name:str):
    nodes = load_data(folder_path)
    nodes_from_doc = [node for node in nodes if node.metadata.get("file_name") == doc_file_name]
    for node in nodes_from_doc:
        print(node)

#data_info("./data_without_questions")
#get_doc_nodes("./data_without_questions", "vracanje investicije.txt")

