import sys

from ssl.EEG_SSL_RP_Dataset import EEG_SSL_Dataset

def test():
    test = EEG_SSL_Dataset("sleep_cassette", 3, 3)
    get = test.__getitem__(69)
    
    
    
    index = test[69]
    #sliced = test[69:420]
    print("\n\n\nget: ")
    print(get)
    print("\n\n\nindex: ")
    print(index)
   # print("sliced: " + sliced)


if __name__ == '__main__':
   test()