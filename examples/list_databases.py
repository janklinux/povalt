import pymongo

# db_con = pymongo.MongoClient(host='numphys.org', port=27017, ssl=True,
#                              ssl_ca_certs='/home/jank/ssl/numphys/ca.crt',
#                              ssl_certfile='/home/jank/ssl/numphys/client.pem')

db_con = pymongo.MongoClient(host='195.148.22.179', port=27017, ssl=False)

db_con.admin.authenticate('jank', 'mongo')

print(db_con.list_database_names())

quit()

new_db = db_con.pot_train
new_db.authenticate('jank', 'b@sf_mongo')
new_coll = new_db['platinum_additions']

print(new_coll.estimated_document_count())

quit()

db_con.pot_train.command('createUser', 'jank', pwd='b@sf_mongo', roles=['readWrite'])
db_con.pot_train.authenticate('jank', 'b@sf_mongo')

# new_coll.delete_one({'junk': 'mail'})
# new_coll.insert_one({'junk': 'mail'})
