#!/usr/bin/env python3

# import dependencies
import json,sys,os,socket,multiprocessing,pickle #,rospy
from rsaServer import *
from rsaClient import *
from euclidInverse import *
from gcdEuclid import *
from primality import *
from random import *
from sys import *

# print(os.listdir("./")) # curr directory

def readFile(path):
    with open(path,"rt") as file:
        content = file.read()
    return content

def readSplitFile(path):
    return readFile(path).split()

def writeFile(path,content):
    output = open(path,"w")
    output.write(content)
    output.close()

class RSASecure(object):
    currVals = {}
    def __init__(self,numBits,fileName):
        self.numBits = numBits
        self.fileName = fileName
    # finds two random prime numbers P,Q with the correct number of bits
    # - adds to currVals dictionary the P,Q values
    def determinePQ(self):
        foundP = foundQ = False
        while not foundP:
            primeP = getrandbits(self.numBits)
            if isPrime(primeP):
                self.currVals["Pprime"] = primeP
                foundP = True
        while not foundQ:
            primeQ = getrandbits(self.numBits)
            if isPrime(primeQ):
                self.currVals["Qprime"] = primeQ
                foundQ = True

    # iterates until valid e,d values used with lambda
    def determineEDKeys(self,lambdaN,modulusPQ):
        dDetermined = eDetermined = False
        while not dDetermined:
            while not eDetermined:
                e = randint(2,lambdaN-1)
                if gcd(e,lambdaN)==1:
                    self.currVals["e"] = e
                    self.currVals["PublicKey"] = (modulusPQ,e)
                    eDetermined = True
            d = multInverse(e,lambdaN)
            if d != None:
                self.currVals["d"] = d
                self.currVals["PrivateKey"] = (modulusPQ,d)
                dDetermined = True

    # generate key using rsa algorithm with inverse module
    def generateKey(self):
        foundP = foundQ = False
        self.determinePQ()
        # module of PQ
        p,q = self.currVals["Pprime"],self.currVals["Qprime"]
        modulusPQ = p*q
        self.currVals["modulusPQ"] = modulusPQ
        # finding lambda(n)
        lambdaN = (p-1)*(q-1)
        # random selection to determine d
        self.determineEDKeys(lambdaN,modulusPQ)
        # writing to fileName
        self.dumpJson(self.fileName)

    # writes currVals (keys) to a json file
    def dumpJson(self,fileName):
        try:
            outputFile = open(fileName,"wt")
            json.dump(self.currVals,outputFile)
            outputFile.close()
        except Exception as e:
            print("Exception {} occured".format(repr(e)))
    # sends key to client while server running
    def distributeKey(self):
        try:
            # ./rsaServer.py run prior to running rsaKey.py
            mainClient(str(self.currVals))
        except Exception as e:
            print("Error: %s"%str(e))
    # converts plaintxt from text file into ciphertext 
    # where characters converted into ascii values & 
    # val V V^e(mod phi)
    def encryptKey(self,plaintxt,ciphertxt):
        plainTxt = readFile(plaintxt)
        publicKey = self.currVals["PublicKey"]
        encrypted = ""
        for c in plainTxt:
            # converting into cryptotext T as T**e (mod phi)
            msgAddition = str(pow(ord(c),publicKey[1],publicKey[0]))+'\n'
            encrypted += msgAddition
        writeFile(ciphertxt,encrypted)
    # descrypts the key using the private key pair
    def decryptKey(self,encryptxt,decryptxt):
        encryptedTxt = readSplitFile(encryptxt)
        privateKey = self.currVals["PrivateKey"]
        decrypted = ""
        for part in encryptedTxt:
            # converting to normal text S as S**d (mod phi)
            msg = chr(pow(int(part),privateKey[1],privateKey[0]))
            decrypted += msg
        writeFile(decryptxt,decrypted)
    # getter method to obtain currVals from the class
    def getKeys(self):
        return self.currVals

def main():
    try:
        assert(len(sys.argv) == 3)
        numBits,fileName = int(sys.argv[1]),sys.argv[2]
        assert(numBits >= 11)
        rsa = RSASecure(numBits,fileName)
        rsa.generateKey()
        rsa.distributeKey()
        rsa.encryptKey("plain.txt","encrypted.txt")
        rsa.decryptKey("encrypted.txt","decrypted.txt")
    except Exception as e:
        sys.exit("Error: {}".format(repr(e)))

if __name__ == "__main__":
    main()
