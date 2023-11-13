from bertopic import BERTopic

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)
