import sys

from ssl.EEG_SSL_RP_Dataset import EEG_SSL_Dataset

def test():
    test = EEG_SSL_Dataset("test", 3, 3)
    get = test.__getitem__(10)
    
    
    
    index = test[10]
    #sliced = test[69:420]
    print("\n\n\nget: ")
    print(get)
    print("\n\n\nindex: ")
    print(index)
   # print("sliced: " + sliced)

    batch = test.get_batch(4)
    print("\n\n\nbatch:")
    print(batch)


if __name__ == '__main__':
   test()

#AC :)