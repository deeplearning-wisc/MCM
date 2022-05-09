pairs = [
('acorn', 'n12267677'),
('airliner', 'n02690373'),
('ambulance', 'n02701002'),
('american_alligator', 'n01698640'),
('banjo', 'n02787622'),
('barn', 'n02793495'),
('bikini', 'n02837789'),
('digital_clock', 'n03196217'),
('dragonfly', 'n02268443'),
('dumbbell', 'n03255030'),
('forklift', 'n03384352'),
('goblet', 'n03443371'),
('grand_piano', 'n03452741'),
('hotdog', 'n07697537'),
('hourglass', 'n03544143'),
('manhole_cover', 'n03717622'),
('mosque', 'n03788195'),
('nail', 'n03804744'),
('parking_meter', 'n03891332'),
('pillow', 'n03938244'),
('revolver', 'n04086273'),
('rotary_dial_telephone', 'n03187595'),
('schooner', 'n04147183'),
('snowmobile', 'n04252077'),
('soccer_ball', 'n04254680'),
('stingray', 'n01498041'),
('strawberry', 'n07745940'),
('tank', 'n04389033'),
('toaster', 'n04442312'),
('volcano', 'n09472597')
]
pairs = dict(pairs)
sorted_pairs = {k: v for k, v in sorted(pairs.items(), key=lambda item: item[1])}
# all_ids = [idx for (name, idx) in pairs]
# names = [name for (name, idx) in pairs]
names = list(sorted_pairs.keys())
print(names)
# with open ('./class_list11.txt', 'w') as f:
#     for id in all_ids:
#         f.write(id +'\n')
