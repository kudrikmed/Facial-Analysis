import telebot
from FaceCropper import FaceCropper
from FacialFeaturesExtractor import FacialFeaturesExtractor
import os.path
import json
import time
import logging


def init_bot():
    f = open('telegram_token.json')
    data = json.load(f)
    telbot = telebot.TeleBot(data["telegram_token"])
    f.close()
    return telbot


def get_text():
    f = open('text.json', encoding="utf-8")
    data = json.load(f)
    f.close()
    return data


bot = init_bot()
text = get_text()
path = 'D://photos/'


@bot.message_handler(content_types='text')
def get_user_text(message):
    if message.from_user.language_code in ['ru', 'en']:
        language = message.from_user.language_code
    else:
        language = 'ru'
    bot.send_message(message.chat.id, text["Hello"][language], parse_mode='html')


@bot.message_handler(content_types='photo')
def get_user_photo(message):
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    if not os.path.exists(path + message.photo[1].file_id):
        os.makedirs(path + message.photo[1].file_id)
        current_path = path + message.photo[1].file_id
        with open(os.path.join(current_path, 'info.txt'), 'a+') as f:
            f.write(str(message))
    src = os.path.join(path + message.photo[1].file_id, message.photo[1].file_id + '.jpg')
    if message.from_user.language_code in ['ru', 'en']:
        language = message.from_user.language_code
    else:
        language = 'ru'
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)
        bot.reply_to(message, text["PhotoAdded"][language])
    fc = FaceCropper(src)
    fc.extract_face_from_image()
    landmarks = fc.get_face_landmarks()
    ffe = FacialFeaturesExtractor(landmarks)

    # mouth cant
    ffe.draw_lip_line_cant(show=False, save_path=current_path)
    bot.send_photo(message.chat.id, photo=open(os.path.join(current_path, 'lips_cant.png'), 'rb'))
    bot.send_message(message.chat.id, f"{text['MouthCornerAngleTheory'][language]}",
                     parse_mode='html')
    bot.send_message(message.chat.id, f"{text['LeftMouthCornerAngle'][language]}: {ffe.get_mouth_corner_tilt()[0]}°, "
                                      f"{text['RightMouthCornerAngle'][language]}: {ffe.get_mouth_corner_tilt()[1]}°",
                     parse_mode='html')
    if (ffe.get_mouth_corner_tilt()[0] + ffe.get_mouth_corner_tilt()[1]) / 2 > 0:
        bot.send_message(message.chat.id, f"{text['MouthCornerAnglePositive'][language]}",
                         parse_mode='html')
    else:
        bot.send_message(message.chat.id, f"{text['MouthCornerAngleNegative'][language]}",
                         parse_mode='html')

    # lips ratio
    ffe.draw_lips_ratio(show=False, save_path=current_path)
    bot.send_photo(message.chat.id, photo=open(os.path.join(current_path, 'lips_ratio.png'), 'rb'))
    bot.send_message(message.chat.id, f"{text['LipsRatioTheory'][language]}",
                     parse_mode='html')
    bot.send_message(message.chat.id, f"{text['LipsRatio'][language]}: {ffe.get_lip_ratio()}",
                     parse_mode='html')
    lips_ratio = float(ffe.get_lip_ratio().partition(':')[2])
    if lips_ratio > 1.6:
        bot.send_message(message.chat.id, f"{text['SmallUpperLip'][language]}",
                         parse_mode='html')
    elif lips_ratio < 1:
        bot.send_message(message.chat.id, f"{text['BigUpperLip'][language]}",
                         parse_mode='html')
    else:
        bot.send_message(message.chat.id, f"{text['IdealLips'][language]}",
                         parse_mode='html')

    # medial eyebrow tilt
    ffe.draw_medial_eyebrow_tilt(show=False, save_path=current_path)
    bot.send_photo(message.chat.id, photo=open(os.path.join(current_path, 'medial_eyebrow_tilt.png'), 'rb'))
    bot.send_message(message.chat.id, f"{text['MedialEyebrowTiltTheory'][language]}",
                     parse_mode='html')
    bot.send_message(message.chat.id, f"{text['LeftMedialEyebrowTilt'][language]}: {ffe.get_medial_eyebrow_tilt()[0]}°, "
                                      f"{text['RightMedialEyebrowTilt'][language]}: {ffe.get_medial_eyebrow_tilt()[1]}°",
                     parse_mode='html')
    if (ffe.get_medial_eyebrow_tilt()[0] + ffe.get_medial_eyebrow_tilt()[1]) / 2 > 25:
        bot.send_message(message.chat.id, f"{text['BrowMedialTiltBig'][language]}",
                         parse_mode='html')
    elif (ffe.get_medial_eyebrow_tilt()[0] + ffe.get_medial_eyebrow_tilt()[1]) / 2 < 15:
        bot.send_message(message.chat.id, f"{text['BrowMedialTiltLow'][language]}",
                         parse_mode='html')
    else:
        bot.send_message(message.chat.id, f"{text['BrowMedialTiltIdeal'][language]}",
                         parse_mode='html')

    # eyebrow apex projection
    ffe.draw_brow_apex_projection(show=False, save_path=current_path)
    bot.send_photo(message.chat.id, photo=open(os.path.join(current_path, 'brow_apex_projection.png'), 'rb'))
    bot.send_message(message.chat.id, f"{text['BrowApexProjectionTheory'][language]}",
                     parse_mode='html')
    bot.send_message(message.chat.id, f"{text['LeftBrowApexProjection'][language]}: {ffe.get_brow_apex_ratio()[0]}, "
                                      f"{text['RightBrowApexProjection'][language]}: {ffe.get_brow_apex_ratio()[1]}",
                     parse_mode='html')
    if (ffe.get_brow_apex_ratio()[0] + ffe.get_brow_apex_ratio()[1]) / 2 > 1:
        bot.send_message(message.chat.id, f"{text['BrowApexIdeal'][language]}",
                         parse_mode='html')
    elif (ffe.get_brow_apex_ratio()[0] + ffe.get_brow_apex_ratio()[1]) / 2 < 0.7:
        bot.send_message(message.chat.id, f"{text['BrowApexLow'][language]}",
                         parse_mode='html')
    else:
        bot.send_message(message.chat.id, f"{text['BrowApexNormal'][language]}",
                         parse_mode='html')

    # upper lip ratio
    ffe.draw_upper_lip_ratio(show=False, save_path=current_path)
    bot.send_photo(message.chat.id, photo=open(os.path.join(current_path, 'upper_lip_ratio.png'), 'rb'))
    bot.send_message(message.chat.id, f"{text['UpperLipRatioTheory'][language]}",
                     parse_mode='html')
    bot.send_message(message.chat.id, f"{text['UpperLipRatio'][language]}: {ffe.get_upper_lip_ratio()}",
                     parse_mode='html')
    upper_lip_ratio = float(ffe.get_upper_lip_ratio().partition(':')[2])
    if upper_lip_ratio > 2.2:
        bot.send_message(message.chat.id, f"{text['BigWhiteUpperLip'][language]}",
                         parse_mode='html')
    elif upper_lip_ratio < 1.8:
        bot.send_message(message.chat.id, f"{text['SmallWhiteUpperLip'][language]}",
                         parse_mode='html')
    else:
        bot.send_message(message.chat.id, f"{text['IdealWhiteUpperLip'][language]}",
                         parse_mode='html')

    # intercanthal tilt
    ffe.draw_canthal_tilt(show=False, save_path=current_path)
    bot.send_photo(message.chat.id, photo=open(os.path.join(current_path, 'canthal_tilt.png'), 'rb'))
    bot.send_message(message.chat.id, f"{text['CanthalTiltTheory'][language]}",
                     parse_mode='html')
    bot.send_message(message.chat.id, f"{text['LeftCanthalTilt'][language]}: {ffe.get_canthal_tilt()[0]}°, "
                                      f"{text['RightCanthalTilt'][language]}: {ffe.get_canthal_tilt()[1]}°",
                     parse_mode='html')
    if (float(ffe.get_canthal_tilt()[0]) + float(ffe.get_canthal_tilt()[0])) / 2 > 7:
        bot.send_message(message.chat.id, f"{text['CanthalTiltIdeal'][language]}",
                         parse_mode='html')
    elif (float(ffe.get_canthal_tilt()[0]) + float(ffe.get_canthal_tilt()[0])) / 2 < 4:
        bot.send_message(message.chat.id, f"{text['CanthalTiltLess'][language]}",
                         parse_mode='html')
    else:
        bot.send_message(message.chat.id, f"{text['CanthalTiltAverage'][language]}",
                         parse_mode='html')

    # bigonial-bizygomatic ratio
    ffe.draw_bigonial_bizygomatic_ratio(show=False, save_path=current_path)
    bot.send_photo(message.chat.id, photo=open(os.path.join(current_path, 'bigonial_bizygomatic_ratio.png'), 'rb'))
    bot.send_message(message.chat.id, f"{text['BigonialBizygomaticRatioTheory'][language]}",
                     parse_mode='html')
    bot.send_message(message.chat.id, f"{text['BigonialBizygomaticRatio'][language]}: "
                                      f"{ffe.get_bigonial_bizygomatic_ratio()}",
                     parse_mode='html')
    if float(ffe.get_bigonial_bizygomatic_ratio()) > 0.75:
        bot.send_message(message.chat.id, f"{text['BigonialBizygomaticRatioMore'][language]}",
                         parse_mode='html')
    elif float(ffe.get_bigonial_bizygomatic_ratio()) < 0.70:
        bot.send_message(message.chat.id, f"{text['BigonialBizygomaticRatioLess'][language]}",
                         parse_mode='html')
    else:
        bot.send_message(message.chat.id, f"{text['BigonialBizygomaticRatioIdeal'][language]}",
                         parse_mode='html')


while True:
    try:
        bot.polling(none_stop=True)

    except Exception as e:
        logging.exception(e)

        time.sleep(15)
