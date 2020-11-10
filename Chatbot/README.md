<h1 align=center><div>
<img src="https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/NTU%20Logo.png " width="500" height="175" align="middle">
</div>

<h1 align=center><font color='Blue'>BC3409</font> - 
<font color='red'>AI in Accounting & Finance</font>

<font size = 5>Semester 1, AY2020-21</font>

<br></br>
<font size = 5>Group Project</font>

# FAANG Telegram Bot
Search `@BC3409_FAANG_Bot` in Telegram. 

[comment:] <> bot description: This bot helps one gain quick insight with the 5 FAANG stocks: Facebook, Amazon, Apple, Netflix, Google. Type /start to begin!
[comment:] <> about text: This bot gives quick insights to the top 5 FAANG stocks using live data.

<p align="center">
  <img src="https://raw.githubusercontent.com/tonyngmk/bc3409project/main/Chatbot/readme_pictures/faangBotPicture.png" />
</p>

**Highlights in Project:**
- Using a full programming language (Python) as backend for Telegram chatbot
- Using yfinance to query live financial data periodically
- Delivering quick insights of both the latest values and plots of both prices and recommendations
- Directly using ML model at backend to predict next 30 days of data for each FAANG stock

*Created for Group Project Assignment in BC3409 - AI in Accounting and Finance AY2020/21 Sem 1.*

## Commands
1. **/start** - Start conversation
2. **/help** - See all commands

## Diagram

### /start
<p align="center">
  <img src="https://raw.githubusercontent.com/tonyngmk/msba_bot/master/diagrams/msba_bot_Start.png" />
</p>


## Replication instructions

Section 1 and 2 describes the backend code of the chatbot. Skip to section 3 to run bot (bot should still be running on Amazon EC2 VM)

### 1. Python-Telegram

We will be using the **telegram** library in Python to create our chatbot.

	python3 -m pip install --user python-telegram-bot

Thereafter, just edit along **bot.py** file and execute it. The python script must continually run for the bot to work. 
To do so, one can run it perpetually using a cloud virtual machine, e.g. AWS EC2, Google Compute Engine, etc. 

### 2. Yahoo Finance

The **yfinance** library allows users to query live financial information of stocks.

As we will be running on AWS EC2 Linux AMI, we can use the *repeat* command on linux to continually update the dataset in regular intervals, such as 86400 seconds (1 day).


### 3. Running

I've tried running on free tier t2 micro and the CPU Credit Usage for 3 bots and is near negligible, so it should be essentially free as well.

<p align="center">
  <img src="https://raw.githubusercontent.com/tonyngmk/my-stoic-telebot/master/cpu_cred_usage.png" />
</p>

In essence, clone this repo and run **bot.py**. As mentioned, bot.py must continually run for chatbot to work.

##### Dump of codes to get it hosted on AWS EC2 Linux2 AMI:

	sudo yum update -y 

	sudo amazon-linux-extras install python3.8

	alias python3='/usr/bin/python3.8'

	python3 --version

	sudo yum install git -y

	sudo yum -y install python3-pip
	
	python3 -m pip install --upgrade pip --user

	git clone https://github.com/tonyngmk/bc3409project.git

	cd bc3409project
	
	cd Chatbot

	chmod 755 ./bot.py

	python3 -m pip install --user python-telegram-bot

	python3 -m pip install tabulate --user
	
	python3 -m pip install --user pandas
	
	python3 -m pip install --user matplotlib
	
	python3 -m pip install --user sklearn
	
	python3 -m pip install --no-cache-dir tensorflow --upgrade --user 
	
	python3 -m pip install --user keras

	screen

	ctrl + a + c (create new screen)

	ctrl + a + n (switch screens)

	python3 bot.py
	
	while sleep 86400; do python3 getData.py; done
	
### Note

This git repo does not contain certain sensitive files (credentials) which has been excluded in .gitignore. In case you are reusing the script, store your:
- Telegram bot's API as **botapi.txt**
