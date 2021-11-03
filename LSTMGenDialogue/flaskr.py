from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for, jsonify, \
    make_response, abort

from ChatbotWeb import *

app = Flask(__name__)
app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.from_envvar('FLASKR_SETTINGS', silent=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

chat_bot_conversations = []


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return 'About Us'

@app.route('/lstm_chatbot_reply', methods=['POST', 'GET'])
def lstm_chatbot_reply():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            chat_bot_conversations.append('YOU: ' + sent)
            #reply = chat_bot_conversations.reply(sent)
            reply = chatbot_response(sent)
            chat_bot_conversations.append('BOT: ' + reply)
    return render_template('lstm_chatbot_reply.html', conversations=chat_bot_conversations)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


def main():
 
    app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    main()
