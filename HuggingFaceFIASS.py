from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings

# Инициализация HuggingFaceInstructEmbeddings
model_name = "hkunlp/instructor-large"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf_embedding = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Встраивание и индексация всех документов с использованием FAISS
db = FAISS.from_texts(all_docs, hf_embedding)

# Сохранение индексаированных данных локально
db.save_local("faiss_AiDoc")