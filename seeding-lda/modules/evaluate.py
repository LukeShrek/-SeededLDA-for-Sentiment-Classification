from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# def cal_comparative_stc_prf()



def cal_sentiment_prf(f1, num_of_aspect, verbal=False):
    f1_score = '{}, {}, {}, {}, {}'.format(f1[0], f1[1], f1[2], f1[3], f1[4])
    macro = sum(f1) / num_of_aspect
    title = 'price,service,safety,quality,ship,authenticity'
    title = 'staff service, room standard, food, location price, facilities'
    if verbal:
        print(title)
        print(f1_score)
        print(macro)


    # output = _p + ', ' + micro_p + ', ' + macro_p + '\n' + _r + '\n' + _f1
    output = ''
    outputs = title + output
    return outputs

