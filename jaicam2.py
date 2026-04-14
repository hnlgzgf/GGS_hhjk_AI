import cv2
import ctypes
import tisgrabber as tis
ic = ctypes.cdll.LoadLibrary("./tisgrabber_x64.dll")
tis.declareFunctions(ic)
class list_device(object):
    def __init__(self):
        ic.IC_InitLibrary(0)
        self.devicecount = ic.IC_GetDeviceCount()
        self.grabbers = []
        self.name=[]
        for i in range(0, self.devicecount):
            print("Device {}".format(tis.D(ic.IC_GetDevice(i))))
            self.uniquename = tis.D(ic.IC_GetUniqueNamefromList(i))
            print("Unique Name : {}".format(self.uniquename))
            self.g = ic.IC_CreateGrabber()
            ic.IC_OpenDevByUniqueName(self.g, tis.T(self.uniquename))
            self.name.append(self.uniquename)
            self.grabbers.append(self.g)
        print(self.name)
        print(self.grabbers)

    def select_device_demo(self,num):

        grabber=self.grabbers[num]
        if (ic.IC_IsDevValid(grabber)):
            ic.IC_SetVideoFormat(grabber, tis.T("RGB32 (1216x1024)"))
            ic.IC_SetFrameRate(grabber, ctypes.c_float(25.0))  # set FrameRate
            ic.IC_StartLive(grabber, 1)
            ic.IC_MsgBox(tis.T("Click OK to stop"), tis.T("Simple Live Video"))
            ic.IC_StopLive(grabber)
        else:
            ic.IC_MsgBox(tis.T("No device opened"), tis.T("Simple Live Video"))
        ic.IC_ReleaseGrabber(grabber)
    def all_device_demo(self):
        for grabber in self.grabbers:
            if (ic.IC_IsDevValid(grabber)):
                ic.IC_SetVideoFormat(grabber, tis.T("RGB32 (1216x1024)"))
                ic.IC_SetFrameRate(grabber, ctypes.c_float(25.0)) # set FrameRate
                ic.IC_StartLive(grabber, 1)
        ic.IC_MsgBox(tis.T("Stop'em all!"), tis.T("Live Video"))

        for grabber in self.grabbers:
            if (ic.IC_IsDevValid(grabber)):
                ic.IC_StopLive(grabber)

        for grabber in self.grabbers:
            if (ic.IC_IsDevValid(grabber)):
                ic.IC_ReleaseGrabber(grabber)

# def eval_clarity(image):
#     return cv2.Laplacian(image, cv2.CV_64F).var()
if __name__ == '__main__':

    device=list_device()
    #device.select_device_demo(2)#指定打开一台相机
    device.all_device_demo()#打开
