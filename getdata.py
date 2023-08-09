from pm4py.objects.log.importer.xes import importer as xes_importer
import numpy as np

def readfile(filename):
    variant = xes_importer.Variants.ITERPARSE
    log = xes_importer.apply(filename, variant=variant)

    print("Total events:", len(log))

    total_events = sum(len(trace) for trace in log)
    print("Total events:", total_events)

    print("events number in the first trace：", len(log[0]))
    print("the first trace:", log[0][0])

    print("keyword", dict(log[0].attributes).keys())
    print(dict(log[0][0]).keys())
    print(dict(log[1][0]).keys())
    return get_feature(log)

def get_feature(log):
    event_log = log

    resource_list = []
    for k in range(len(event_log)):
        trace1 = []
        for j in range(len(event_log[k])):
            if 'Producer code' in event_log[k][j]:  # hospital: 'Producer code'
                # event_log[k][j]['org:resource']
                trace1.append(event_log[k][j]['Producer code'])  # 'org:resource'

        resource_list.append(trace1)

    all_elements = set()
    for l in resource_list:
        all_elements.update(l)
    col_num = len(all_elements)

    resource_matrix = [[0] * col_num for _ in range(len(resource_list))]

    for i, l in enumerate(resource_list):
        for j, e in enumerate(all_elements):
            resource_matrix[i][j] = l.count(e)

    resource_arr = np.array(resource_matrix)
    print("Resource:", resource_arr.shape)

    activity_list = []
    for k in range(len(event_log)):
        trace1 = []
        for j in range(len(event_log[k])):
            if 'concept:name' in event_log[k][j]:
                trace1.append(event_log[k][j]['concept:name'])
            else:
                trace1.append('0')

        activity_list.append(trace1)

    all_elements = set()
    for l in activity_list:
        all_elements.update(l)
    col_num = len(all_elements)

    activity_matrix = [[0] * col_num for _ in range(len(activity_list))]

    for i, l in enumerate(activity_list):
        for j, e in enumerate(all_elements):
            activity_matrix[i][j] = l.count(e)

    activity_arr = np.array(activity_matrix)
    print("Activity:", activity_arr.shape)

    transition_list = []
    for k in range(len(activity_list)):
        trace1 = []
        for j in range(len(activity_list[k]) - 1):
            trace1.append(activity_list[k][j] + '-' + activity_list[k][j + 1])
        transition_list.append(trace1)

    # print(len(transition_list))

    all_elements = set()
    for l in transition_list:
        all_elements.update(l)
    col_num = len(all_elements)

    transition_matrix = [[0] * col_num for _ in range(len(transition_list))]

    for i, l in enumerate(transition_list):
        for j, e in enumerate(all_elements):
            transition_matrix[i][j] = l.count(e)

    # 打印矩阵
    transition_arr = np.array(transition_matrix)
    print("Transition:", transition_arr.shape)

    all_matrix = [resource_matrix, activity_matrix, transition_matrix]
    print("Overview:", len(all_matrix))

    all_arr = np.array([resource_arr.T, activity_arr.T, transition_arr.T], dtype='object')
    np.savez('./hospital_overview', all_arr)


def main():
    usage = """\
        usage:
               getdata.py [--f value]
        options:
                -f filename -- Log file, default Hospital_log.xes
        """

    import getopt, sys

    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:")
        if len(opts) == 0:
            print(usage)
            return

        global filename
        filename = "Hospital_log.xes"

        for opt, value in opts:
            if opt == '--f':
                filename = value

        print("--------------------------------------------------------------")
        print(" Log: ", filename)
        print("--------------------------------------------------------------")

        readfile(filename)

    except getopt.GetoptError:
        print(usage)
    except SyntaxError as error:
        print(error)
        print(usage)
    return 0


if __name__ == '__main__':
    main()