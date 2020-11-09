#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler
from telegram import (ReplyKeyboardMarkup, ReplyKeyboardRemove)
import logging
import json
from datetime import datetime, timedelta
import os
from oauth2client.service_account import ServiceAccountCredentials
from pprint import pprint
import pandas as pd
import re
import pickle
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import datetime

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)
logger = logging.getLogger(__name__)

lastYear = datetime.datetime.now() - datetime.timedelta(days=365) # Dynamic date of today - 365 days
df, a, b, c, dfRecommendations = range(5) # dataset
FAANG, FACEBOOK, AMAZON, APPLE, NETFLIX, GOOGLE, REPEATOREXIT = range(7) # states

def getData():
    # stocks = yf.Tickers(['FB','AMZN','NFLX','GOOGL','AAPL'])
    # time_period = 'max'#can change this
    # df = stocks.history(period = time_period)
    # df = df['Close']
    # df = df.reset_index()
    # df = df[df.notna().all(axis = 1)]
    # df.to_csv("prices.csv", index=False)
    df = pd.read_csv("prices.csv")
    df["Date"] = pd.to_datetime(df["Date"]) # convert string to datetime format
    stocks = ['FB','AMZN','NFLX','GOOGL','AAPL']
    scaler = StandardScaler()
    df[stocks] = scaler.fit_transform(df[stocks])
    train_size = int(df.shape[0]*0.80)
    val_size = int(df.shape[0]*0.90)
    a = df.iloc[:train_size, :]
    b = df.iloc[train_size:val_size, :]
    c = df.iloc[val_size:, :]
    a=a.drop("Date", axis =1)
    b=b.drop("Date", axis =1)
    c=c.drop("Date", axis =1)
    
    # stocks = ['FB','AMZN','NFLX','GOOGL','AAPL']
    # masterlist = []
    # for each in stocks:
        # x = yf.Ticker(each)
        # y = x.recommendations
        # y['stock_symbol'] = each
        # y = y.reset_index()
        # masterlist.append(y)
    # dfRecommendations = pd.concat(masterlist,axis= 0)
    # dfRecommendations["Date"] = pd.to_datetime(dfRecommendations["Date"]) # convert string to datetime format
    # dfRecommendations.to_csv('recommendations.csv',index = False)
    # dfRecommendations
    dfRecommendations = pd.read_csv("recommendations.csv")# Read from cache to reduce time taken
    return df, a, b, c, dfRecommendations

def start(update, context):
    user = update.message.from_user
    logger.info("User {} has started to use FAANG bot".format(user.first_name, update.message.text))
    # context.bot.send_message(chat_id=update.effective_chat.id, text="⌛⌛⌛ Grabbing data, please allow approximately 5s!")
    global df, a, b, c, dfRecommendations
    df, a, b, c, dfRecommendations = getData()
    context.bot.send_message(chat_id=update.effective_chat.id, text="Connected! ✅ Latest data is at {}".format(df.iloc[-1].Date.strftime('%d-%m-%Y')))
    context.bot.send_message(chat_id=update.effective_chat.id, text="👋👋 Hey there! Welcome to the the FAANG Stocks Chatbot! 🤖")
    context.bot.send_message(chat_id=update.effective_chat.id, text="👩‍💻👩‍💻👩‍💻 **FAANG** refers to the five most popular and best-performing US tech companies: _Facebook, Amazon, Apple, Netflix and Alphabet_.", parse_mode=telegram.ParseMode.MARKDOWN)
    kb = [[telegram.KeyboardButton('Facebook')],
          [telegram.KeyboardButton('Amazon')],
          [telegram.KeyboardButton('Apple')],
          [telegram.KeyboardButton('Netflix')],
          [telegram.KeyboardButton('Google')]
          ]
    kb_markup = telegram.ReplyKeyboardMarkup(kb, one_time_keyboard=True)
    update.message.reply_text("🚀 To continue, please kindly select one of the FAANG stock 🚀\n\n Alternatively, type /No anywhere to cancel.\n\n_Do scroll down for your keyboard options_", reply_markup=kb_markup, parse_mode=telegram.ParseMode.MARKDOWN)
    return FAANG

def faangCont(update, context, stock):
    user = update.message.from_user
    logger.info("User {} has selected to 'Learn More' for {}".format(user.first_name, stock))
    context.bot.send_message(chat_id=update.effective_chat.id, text="🔥 Thank you for your interest in _{}_! 🔥".format(stock), reply_markup=ReplyKeyboardRemove(), parse_mode=telegram.ParseMode.MARKDOWN)
    context.bot.send_message(chat_id=update.effective_chat.id, text="📚 We have a plethora of resources available to help you in your analysis.\n\n_Do scroll down for your keyboard options_", parse_mode=telegram.ParseMode.MARKDOWN)
    kb = [[telegram.KeyboardButton('💰 Stocks Prices')],
          [telegram.KeyboardButton('❓ Recommendations')],
          [telegram.KeyboardButton('🧠 Prediction')],
          [telegram.KeyboardButton('🏠 Main Menu (FAANG)')]]
    kb_markup = telegram.ReplyKeyboardMarkup(kb, one_time_keyboard=True)
    update.message.reply_text("🚀 To continue, please kindly select one of the options 🚀\n\n Alternatively, type /No anywhere to cancel.", reply_markup=kb_markup)

def facebook(update, context):
    faangCont(update, context, "FB")
    return FACEBOOK
def amazon(update, context):
    faangCont(update, context, "AMZN")
    return AMAZON
def apple(update, context):
    faangCont(update, context, "AAPL")
    return APPLE
def netflix(update, context):
    faangCont(update, context, "NFLX")
    return NETFLIX
def google(update, context):
    faangCont(update, context, "GOOGL")
    return GOOGLE

def getPrice(update, context, stock):
    global df
    user = update.message.from_user
    logger.info("User {} has selected to 'Stocks Prices' for {}".format(user.first_name, stock))
    context.bot.send_message(chat_id=update.effective_chat.id, text='⬇ The latest 10 days of data for {} is being populated.'.format(stock))
    priceTable = df[["Date", stock]].tail(10)
    priceTable["Date"] = priceTable["Date"].dt.date
    priceTable = priceTable.set_index("Date").to_markdown()
    context.bot.send_message(chat_id=update.effective_chat.id, text='<pre>{}</pre>'.format(priceTable), parse_mode=telegram.ParseMode.HTML)
    context.bot.send_message(chat_id=update.effective_chat.id, text='⬇ The latest 365 days of data for {} is being plotted. \n\n⌛⌛⌛ Please allow approximately 10 seconds. '.format(stock))
    ax = df[["Date", stock]].tail(365).set_index("Date").plot(figsize = (30,15))
    ax.tick_params(axis="x", labelsize=20, rotation=45)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_title('Time-series plot of latest 365 days for {}'.format(stock), fontsize = 22)
    ax.set_ylabel('Value', fontsize = 20)
    ax.set_xlabel('Date', fontsize = 20)
    ax.legend(prop={'size': 20})
    plt.savefig('{}PricePlot.png'.format(stock), bbox_inches = 'tight', pad_inches = 0.1)
    context.bot.send_photo(chat_id=update.effective_chat.id, photo = open("{}PricePlot.png".format(stock), 'rb'))
    kb = [[telegram.KeyboardButton('➡ Learn More ({})'.format(stock))],
          [telegram.KeyboardButton('🏠 Main Menu (FAANG)')],
          [telegram.KeyboardButton('/No')]]
    kb_markup = telegram.ReplyKeyboardMarkup(kb, one_time_keyboard=True)
    update.message.reply_text("🚀 That is all. Do you want to continue learning more about {} or return to main menu to all FAANG stocks? 🚀\n\n Alternatively, type /No anywhere to cancel.".format(stock), reply_markup=kb_markup)

def facebookPrice(update, context):
    stock = 'FB'
    getPrice(update, context, stock)
    return REPEATOREXIT
def amazonPrice(update, context):
    stock = 'AMZN'
    getPrice(update, context, stock)
    return REPEATOREXIT
def applePrice(update, context):
    stock = 'AAPL'
    getPrice(update, context, stock)
    return REPEATOREXIT
def netflixPrice(update, context):
    stock = 'NFLX'
    getPrice(update, context, stock)
    return REPEATOREXIT
def googlePrice(update, context):
    stock = 'GOOGL'
    getPrice(update, context, stock)
    return REPEATOREXIT

def getRec(update, context, stock):
    global dfRecommendations
    user = update.message.from_user
    logger.info("User {} has selected to 'Recommendations' for {}".format(user.first_name, stock))
    recTable = dfRecommendations[(dfRecommendations["stock_symbol"] == stock) & (dfRecommendations["Date"] > lastYear)].sort_values('Date')
    context.bot.send_message(chat_id=update.effective_chat.id, text='⬇ The latest 10 recommendations for {} is being populated.'.format(stock))
    recTableTail = recTable[["Date", "Firm", "To Grade", "From Grade"]].tail(10)
    recTableTail["Date"] = recTableTail["Date"].dt.date
    recTableTail = recTableTail.set_index("Date").to_markdown()
    context.bot.send_message(chat_id=update.effective_chat.id, text='<pre>{}</pre>'.format(recTableTail), parse_mode=telegram.ParseMode.HTML)

def facebookRec(update, context):
    pass
def amazonRec(update, context):
    pass
def appleRec(update, context):
    pass
def netflixRec(update, context):
    pass
def googleRec(update, context):
    pass
    
def facebookPred(update, context):
    pass
def amazonPred(update, context):
    pass
def applePred(update, context):
    pass
def netflixPred(update, context):
    pass
def googlePred(update, context):
    pass

def cancel(update, context):
    user = update.message.from_user
    logger.info("User %s canceled the conversation via /No.", user.first_name)
    update.message.reply_text('Bye! Hope we can talk again soon. You know where to (/start) 😁',
                              reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

def help(update, context):
    user = update.message.from_user
    logger.info("User %s queried for commands via /help", user.first_name)
    context.bot.send_message(chat_id=update.effective_chat.id, text="😃 Don't worry, we're here to help! 😃")
    context.bot.send_message(chat_id=update.effective_chat.id, text='''
*Commands:*
/start - Start the conversation
/join - Join mailing list
/upload - Upload CV for pre-assessment
/help - See this again
''', parse_mode=telegram.ParseMode.MARKDOWN)

def main():
    f = open("botapi.txt")
    TOKEN = f.readlines()[0]
    f.close()
    updater = Updater(TOKEN, use_context=True)
    dispatcher = updater.dispatcher
    
    faang_conv_handler = ConversationHandler(
        entry_points = [CommandHandler('start', start)],
        states={
            FAANG : [MessageHandler(Filters.regex('Facebook'), facebook),
                     MessageHandler(Filters.regex('Amazon'), amazon),
                     MessageHandler(Filters.regex('Apple'), apple),
                     MessageHandler(Filters.regex('Netflix'), netflix),
                     MessageHandler(Filters.regex('Google'), google),
                    ],
            FACEBOOK: [MessageHandler(Filters.regex('💰 Stocks Prices'), facebookPrice),
                       MessageHandler(Filters.regex('❓ Recommendations'), facebookRec),
                       MessageHandler(Filters.regex('🧠 Prediction'), facebookPred),
                       MessageHandler(Filters.regex('🏠 Main Menu \(FAANG\)'), start)
                    ],
            AMAZON: [MessageHandler(Filters.regex('💰 Stocks Prices'), amazonPrice),
                       MessageHandler(Filters.regex('❓ Recommendations'), amazonRec),
                       MessageHandler(Filters.regex('🧠 Prediction'), amazonPred),
                       MessageHandler(Filters.regex('🏠 Main Menu \(FAANG\)'), start)
                    ],
            APPLE: [MessageHandler(Filters.regex('💰 Stocks Prices'), applePrice),
                       MessageHandler(Filters.regex('❓ Recommendations'), appleRec),
                       MessageHandler(Filters.regex('🧠 Prediction'), applePred),
                       MessageHandler(Filters.regex('🏠 Main Menu \(FAANG\)'), start)
                    ],
            NETFLIX: [MessageHandler(Filters.regex('💰 Stocks Prices'), netflixPrice),
                       MessageHandler(Filters.regex('❓ Recommendations'), netflixRec),
                       MessageHandler(Filters.regex('🧠 Prediction'), netflixPred),
                       MessageHandler(Filters.regex('🏠 Main Menu \(FAANG\)'), start)
                    ],
            GOOGLE: [MessageHandler(Filters.regex('💰 Stocks Prices'), googlePrice),
                       MessageHandler(Filters.regex('❓ Recommendations'), googleRec),
                       MessageHandler(Filters.regex('🧠 Prediction'), googlePred),
                       MessageHandler(Filters.regex('🏠 Main Menu \(FAANG\)'), start)
                    ],
            REPEATOREXIT: [MessageHandler(Filters.regex('➡ Learn More \(FB\)'), facebook),
                           MessageHandler(Filters.regex('➡ Learn More \(AMZN\)'), amazon),
                           MessageHandler(Filters.regex('➡ Learn More \(AAPL\)'), apple),
                           MessageHandler(Filters.regex('➡ Learn More \(NFLX\)'), netflix),
                           MessageHandler(Filters.regex('➡ Learn More \(GOOGL\)'), google),
                           MessageHandler(Filters.regex('➡ Learn More \(FB\)'), facebook),
                           MessageHandler(Filters.regex('🏠 Main Menu \(FAANG\)'), start)
            ]
        },
        fallbacks=[CommandHandler('No', cancel)])
    dispatcher.add_handler(faang_conv_handler)  


    # Help
    helpCommand = CommandHandler('help', help)
    dispatcher.add_handler(helpCommand)

    # Launch
    updater.start_polling() # Start locally hosting Bot
    updater.idle()  # Run the bot until you press Ctrl-C or the process receives SIGINT,

    # PORT = int(os.environ.get('PORT', 5000))
    # updater.start_webhook(listen="0.0.0.0",
                          # port=int(PORT),
                          # url_path=TOKEN)
    # updater.bot.setWebhook('https://my-stoic-telebot.herokuapp.com/' + TOKEN)
    
if __name__ == '__main__':
    main()