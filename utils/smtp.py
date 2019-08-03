'''
This module offers a class for sending a mail as notification.
For example you are running a long calculation on a server and want to get updates,
when a specific event is triggered.
Then you import this module, initialize a SMTPNotifier() and call a notify() in your code,
when the event is triggered.

Note: The SMTPNotifier class was written only for Gmail / G Suite accounts.
However you should be able to change the smtpserver in the class to any other server.

Important: The Google Account needs to allow login from 'less secure apps'
You can change these settings for your account here:
https://myaccount.google.com/lesssecureapps
'''

import smtplib
import datetime
import getpass


class SMTPNotifier():
    '''This is a class for sending mails from G Suite / Gmail (like @dida.do) over SMTP easily.

     Parameters
     ----------
     :param receiver_mail_addresses : str or [str]
        mail addresses of everyone who should receive the messages

     :param sender_mail_address (optional) : str
         the mail address of the sender

     :param cc_mail_addresses (optional) : [str]
        the mail addresses for the people in the CC.

    Usage
    -----
    notifier = SMTPNotifier(receiver, sender (optional), cc (optional))
    notifier.notify(message, subject (optional))
    '''

    def __init__(self,
                 receiver_mail_addresses,
                 sender_mail_address='da@dida.do',
                 cc_mail_addresses=[]):

        self.smtpserver = 'smtp.gmail.com:587'
        self.sender_mail_address = sender_mail_address
        self.receiver_mail_addresses = receiver_mail_addresses if isinstance(receiver_mail_addresses, list) else [
            receiver_mail_addresses]
        self.cc_mail_addresses = cc_mail_addresses

        prompt = f'Enter the password for the mail account: "{sender_mail_address}":\n'

        while True:
            self.__password = getpass.getpass(prompt=prompt)

            if self.test_connection():
                break
            else:
                prompt = f'Login failed! Please reenter your password for "{sender_mail_address}":\n'

        print('Login successful!')

    def test_connection(self):
        '''This method checks if the smtpserver accepts the connection

        :return: Bool
            True if it accepts, False if not
        '''
        server = smtplib.SMTP(self.smtpserver)
        server.starttls()
        try:
            server.login(self.sender_mail_address, self.__password)
            server.quit()
            return True
        except smtplib.SMTPAuthenticationError:
            server.quit()
            return False

    def notify(self, message, subject='Updates from ' + str(datetime.datetime.now().strftime('%d.%m.%Y - %H:%M'))):
        '''Sends a mail on the established connection.

        :param message : str
            message to be in the mail

        :param subject : str
            subject of the mail

        :return : {}
            errors that migth occured while sending the mail
        '''
        return send_email(self.sender_mail_address,
                          self.receiver_mail_addresses,
                          self.cc_mail_addresses,
                          subject,
                          message,
                          self.sender_mail_address,
                          self.__password,
                          self.smtpserver)


def send_email(sender_mail_address, receivers_mail_addresses, cc_mail_addresses,
               subject, message,
               login, password,
               smtpserver='smtp.gmail.com:587'):
    '''A high level function to send a mail. Gets called by Notifier.
    Parameters:
    -----------
    :param sender_mail_address : str
        the mail address you want to send from

    :param receivers_mail_addresses : [str]
        the mail addresses that should receive the messages

    :param cc_mail_addresses : [str]
        the mail addresses to be in the cc

    :param subject : str
        the subject of the mail

    :param message : str
        the message you want to send

    :param login : str
        the username to login with the mailserver. Often equivalent to the sender_mail_address

    :param password : str
        password to login at the mailserver

    :param smtpserver : str
        the smtpserver address you want to login with port

    :return : {}
        errors that might occurred while sending the mail
    '''
    header = 'From: %s\n' % sender_mail_address
    header += 'To: %s\n' % ','.join(receivers_mail_addresses)
    header += 'Cc: %s\n' % ','.join(cc_mail_addresses)
    header += 'Subject: %s\n\n' % subject
    message = header + message

    server = smtplib.SMTP(smtpserver)
    server.starttls()
    server.login(login, password)
    problems = server.sendmail(sender_mail_address, receivers_mail_addresses, message.encode("utf8"))
    server.quit()
    return problems
