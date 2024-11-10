import time
import sys
# import datetime
# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console

# Print iterations progress
def printProgressBar(iteration
                    , total
                    , start_time
                    , prefix = ''
                    , suffix = 'Done'
                    , suffix_list = []
                    , decimals = 1
                    , length = 10
                    , fill = '#'
                    , printEnd = "\r"
                    , start_time_str = ""
                    , output_lines = None
                    ):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    if iteration == total: 
        printEnd = "\n"
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    eta_time = eta(start_time, iteration, total)

    # current_dt = datetime.datetime.now()
    # print(' Start Training:'.format(current_dt.strftime("%m-%d %H:%M:%S")))
    print('\r' + start_time_str, '%s|%s| %s%% %s ETA:%s' % (prefix, bar, percent, suffix, eta_time), end = printEnd)
    """
    #suffix_list = suffix_list.split('\n')
    #print(suffix)
    suffix_list = suffix_list + [" "]
    #with output(initial_len=len(suffix_list)+1, interval=0) as output_lines :
    #with output(initial_len=len(suffix), interval=0) as output_lines :
    bar_str = "{}|{}| {}% {} ETA:{}".format(prefix, bar, percent, suffix, eta_time)
    #output_lines[0] = '%s|%s| %s%% %s ETA:%s' % (prefix, bar, percent, "", eta_time))
    output_lines[0] = bar_str
    for i in range(len(suffix_list)) :
        output_lines[i+1] = suffix_list[i]
    """
    # Print New Line on Complete
    #if iteration == total: 
    #    print('\n')


def cal_running_avg_loss(loss, running_avg_loss, decay=0.99):
    if running_avg_loss == 0:
        return loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
        return running_avg_loss


def time_since(t):
    """ Function for time. """
    return time.time() - t


def eta(start, completed, total):
    """ Function returning an ETA. """
    # Computation
    took = time_since(start)
    time_per_step = took / completed
    remaining_steps = total - completed
    remaining_time = time_per_step * remaining_steps
    return user_friendly_time(remaining_time)


def user_friendly_time(s):
    """ Display a user friendly time from number of second. """
    s = int(s)
    if s < 60:
        return "{}s".format(s)

    m = s // 60
    s = s % 60
    if m < 60:
        return "{}m {}s".format(m, s)

    h = m // 60
    m = m % 60
    if h < 24:
        return "{}h {}m {}s".format(h, m, s)

    d = h // 24
    h = h % 24
    return "{}d {}h {}m {}s".format(d, h, m, s)
