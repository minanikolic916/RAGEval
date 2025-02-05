
class FormatedNode:
    def __init__(self, id, file_name, text, score, ret_score):
        self.id = id
        self.file_name = file_name
        self.text = text
        self.score = score
        self.ret_score = ret_score
        
def get_formated_nodes(nodes):
    formated_nodes = []
    for node in nodes:
        formated_node = FormatedNode(
            id = node.id_,
            file_name = node.node.metadata.get("file_name"),
            text = node.node.get_content(),
            score = node.score, 
            ret_score = node.node.metadata.get("retrieval_score")
        )
        formated_nodes.append(formated_node)
    return formated_nodes

def display_ret_nodes(formated_nodes):
    print('-' * 100)
    for node in formated_nodes:
        id = node.id
        file_name = node.file_name
        text = node.text
        score = node.score
        ret_score = node.ret_score
        print(f"\nID: {id}\nFile_name: {file_name}\nText:{text}\nScore:{score}\nRetScore:{ret_score}\n")
        print('-' * 100)

def final_display_context(response):
    formated_nodes = get_formated_nodes(response)
    display_ret_nodes(formated_nodes= formated_nodes)
    context = formated_nodes[0].text
    return context
