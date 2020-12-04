from fireworks import LaunchPad


lpad = LaunchPad(host='195.148.22.179', port=27017, name='atlas_fw', username='jank', password='mongo', ssl=False)

all_jobs = lpad.get_fw_ids(query={'state': 'COMPLETED'})
print(len(all_jobs))

for job in all_jobs:
    ldir = lpad.get_launchdir(job)
    quit()
