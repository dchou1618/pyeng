#/usr/bin/env python3

import module_manager
module_manager.review()

import firebase_admin,datetime,time
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud.firestore_v1beta1 import ArrayRemove, ArrayUnion
# Organization of firebase
# collections (name: students) -
    # documents ('document1')
        # fields (key:value pair)

cred = credentials.Certificate('serviceAccount.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

doc_ref = db.collection(u'users').document(u'alovelace') # accessing document
# level in database
doc_ref.set({
    u'first': u'Ada',
    u'last': u'Lovelace',
    u'born': 1815
}) # sets attributes
#Add another(different) document
doc_ref = db.collection(u'users').document(u'aturing')
doc_ref.set({
    u'first': u'Alan',
    u'middle': u'Mathison',
    u'last': u'Turing',
    u'born': 1912
})

#To read data you can see it online or by using the following code
users_ref = db.collection(u'users')
docs = users_ref.get() # getting users references
#doc.id is the name of the document, doc.to_dict returns all of its attributes
for doc in docs:
    print(u'{} => {}'.format(doc.id, doc.to_dict()))


data = {
    u'name': u'Los Angeles',
    u'state': u'CA',
    u'country': u'USA'
}
# Add a new doc in collection 'cities' with ID 'LA'
db.collection(u'cities').document(u'LA').set(data)
#Set merge = True in case you don't know if a document exists
city_ref = db.collection(u'cities').document(u'BJ')
city_ref.set({
    'capital': True
},merge = True)


#Here are all the types of data you can store in Firestore
data = {
    u'stringExample': u'Hello, World!',
    u'booleanExample': True,
    u'numberExample': 3.14159265,
    u'dateExample': datetime.datetime.now(),
    u'arrayExample': [5, True, u'hello'],
    u'nullExample': None,
    u'objectExample': {
        u'a': 5,
        u'b': True
    }
}
db.collection(u'data').document(u'one').set(data)

#Example of how to create objects and then add them to the database
class Person(object):
    def __init__(self,age):
        self.age = age
    #Call this method when you want to add this object to the database
    def to_dict(self):
        return {u'age':self.age}
    #We'll use this later to create a City object from a document in the database
    def from_dict(document):
        return Person(document[u'age'])
    def __repr__(self):
        return 'Person(%d)'%self.age
#You can set the dictionary containing the objects attributes to the database
me = Person(age = 21)
db.collection(u'users').document(u'Marco').set(me.to_dict())

#To Update information
ref = db.collection(u'users').document(u'Marco')
ref.update({u'age':19})


#To update dictionaries (From the Google Documentation)
# Create an initial document to update
frank_ref = db.collection(u'users').document(u'frank') # getting to frank in
# documents
frank_ref.set({
    u'name': u'Frank',
    u'favorites': {
        u'food': u'Pizza',
        u'color': u'Blue',
        u'subject': u'Recess'
    },
    u'age': 12
})

# Update age and favorite color (which is a nested dictionary, by using the dot-notation)
frank_ref.update({
    u'age': 13,
    u'favorites.color': u'Red'
})

#To Update Arrays
city_ref = db.collection(u'cities').document(u'DC')
city_ref.set({u'regions':['east_coast']})
# Automatically add a new region to the 'regions' array field.
city_ref.update({u'regions': ArrayUnion([u'greater_virginia'])})
# // Automatically remove a region from the 'regions' array field.
#Uncomment this line to remove the east_cost property
#city_ref.update({u'regions': ArrayRemove([u'east_coast'])})


#To delete attributes from a document use .DELETE_FIELD
city_ref = db.collection(u'cities').document(u'BJ')
city_ref.update({
    u'capital': firestore.DELETE_FIELD
})
# DELETE_FIELD automatically deletes field


#This is how you create a Python object from the values in the database
doc_ref = db.collection(u'users').document(u'Marco')
doc = doc_ref.get()
person = Person.from_dict(document = doc.to_dict())
print(person)

#This is how you query, or filter through a collection for specific documents
#.where() can also use u'<', u'>', u'==', u'<=', u'>=', and u'array_contains'
user_collection = db.collection(u'users')
query = user_collection.where(u'age', u'==', 21)


#This is how you "listen" to your documents. First we'll set up a query and then
#begin listening to it with this code from the google cloud website!
# This function continuously reacts to changes in a query you ran.
def on_snapshot(col_snapshot, changes, read_time):
    print(u'Callback received query snapshot.')
    print(u'Current cities in California: ')
    for change in changes:
        #It listens for added, modified, and removed
        if change.type.name == 'ADDED':
            print(u'New city: {}'.format(change.document.id))
        elif change.type.name == 'MODIFIED':
            print(u'Modified city: {}'.format(change.document.id))
        elif change.type.name == 'REMOVED':
            print(u'Removed city: {}'.format(change.document.id))
col_query = db.collection(u'cities').where(u'state', u'==', u'CA')
# Watch the collection query
query_watch = col_query.on_snapshot(on_snapshot)
# [END listen_multiple]
# Creating document
data = {
    u'name': u'San Francisco',
    u'state': u'CA',
    u'country': u'USA',
    u'capital': False,
    u'population': 860000
}
db.collection(u'cities').document(u'SF').set(data)
#Try this code with the forever loop below, if you try acessing this database
#from somewhere else this code should be updating live!
# while(True):
#     pass
time.sleep(1)
query_watch.unsubscribe()
