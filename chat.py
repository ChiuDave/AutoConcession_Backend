def reset_chat():
    conversation_history.clear()
    return jsonify({'status': 'success'})

def chat_endpoint():
    data = request.get_json()
    user_input = data.get('message', '').strip()
    if not user_input:
        return jsonify({'error': 'No message provided.'}), 400

    try:
        is_follow_up = "follow-up" in user_input.lower() or (len(conversation_history) > 0 and not user_input.lower().startswith(('new', 'reset', 'start over')))
        session_context = ""
        enriched_user_input = user_input

        if is_follow_up:
            session_context = "\n".join(
                f"User: {entry['query']}\nAssistant: {entry['response']}"
                for entry in conversation_history[-6:]
            )
            enriched_user_input = f"{session_context}\nUser: {user_input}"

        sql_query = generate_sql_from_input(enriched_user_input, sqlChat)

        print("Query is: ", sql_query)
    
        try:
            conn = sqlite3.connect('data.db')
            filtered_df = pd.read_sql_query(sql_query.content, conn)
            conn.close()

            texts = filtered_df["description"].tolist()
            metadata = filtered_df.drop(columns=["description"]).to_dict(orient="records")
            vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadata)
        except Exception as e:

            texts = df["description"].tolist()
            metadata = df.drop(columns=["description"]).to_dict(orient="records")

            vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadata)


        enriched_query = enriched_user_input if is_follow_up else user_input
        search_results = vectorstore.similarity_search(enriched_query, k=3)

        metadata_results = [sanitize_metadata(result.metadata) for result in search_results]

        formatted_results = "\n".join(
            f"{i+1}. {result.page_content}"
            for i, result in enumerate(search_results)
        )

        response = chain.invoke({
            "query": enriched_user_input,
            "results": formatted_results
        })

        conversation_history.append({
            "query": user_input,
            "response": response.content
        })

        print(f"\nAssistant: {response.content}")

        response_data = {
            'response': response.content,
            'metadata': metadata_results
        }

        return jsonify(response_data)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500