import json



if __name__ == "__main__":

    with open("/home/iraklis/PycharmProjects/Clef_contest/I_O/output/Document/"
              "dev_filled.json", "r") as json_data:
        count = 0
        for line in json_data:
            d = json.loads(line)
            count += 1
            # print("a")
        print(count)