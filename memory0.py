import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.classes.config import Configure

client = weaviate.connect_to_local()

class_name = "Insights"

# Проверка существования класса
existing_classes = client.collections.list_all()
if class_name not in existing_classes:
    # Создание класса
    client.collections.create(
        name=class_name,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="timestamp", data_type=DataType.DATE),
        ],
    )
    print(f"Класс '{class_name}' создан успешно.")
else:
    print(f"Класс '{class_name}' уже существует.")
