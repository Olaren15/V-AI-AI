from datetime import datetime

from flask import Flask, jsonify, request
from flask_expects_json import expects_json

from conversation import Conversation
import uuid

app = Flask(__name__)

conversations = {}

reply_schema = {
    'type': 'object',
    'properties': {
        'message': {'type': 'string'},
        'conversation_id': {'type': 'string'}
    },
    'required': ['message']
}


@app.route('/reply', methods=['POST'])
@expects_json(reply_schema)
def reply():
    data = request.json
    message = data['message']

    if 'conversation_id' not in data:
        conversation_id = uuid.uuid4().hex
        conversations[conversation_id] = Conversation()
    elif data['conversation_id'] not in conversations:
        return jsonify({
            'error': 'Conversation ID does not exist',
            'code': 404
        }), 404
    else:
        conversation_id = data['conversation_id']

    answer = conversations[conversation_id].reply(message)

    return jsonify({
        'answer': answer,
        'conversation_id': conversation_id
    })


@app.after_request
def clean_unused_conversations(response):
    now = datetime.now()
    to_pop = []

    for conversation_id, conversation in conversations.items():
        delta = now - conversation.last_interaction
        if delta.total_seconds() > 30:  # 15 minutes
            to_pop.append(conversation_id)

    for item in to_pop:
        conversations.pop(item)

    return response


@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        'error': error.description.message,
        'status': 400
    }), 400


@app.errorhandler(500)
def server_error(error):
    return jsonify({
        'error': 'an error occured',
        'status': 500
    }), 500


if __name__ == '__main__':
    from waitress import serve

    serve(app, host='0.0.0.0', port=5000)
