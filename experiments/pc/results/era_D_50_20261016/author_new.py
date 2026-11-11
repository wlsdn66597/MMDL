"""Explicit image-grounded common content and natural paraphrases for 45 new drafts.
Domain helpers only serialize author-inspected facts; they do not infer image content.
"""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA={}
def add(sid,e,r,a,parts,tag,note='',disagreement=None):
    g=' '.join(t for _,t in parts)
    units=[]
    for k,meaning in [('e',e),('r',r),('a',a)]:
        spans=[t for keys,t in parts if k in keys]
        assert spans,(sid,k)
        # Preserve a contiguous supporting excerpt, including intervening prose if needed.
        start=g.index(spans[0]);end=g.index(spans[-1])+len(spans[-1])
        units.append([meaning,g[start:end],meaning])
    DATA[sid]=dict(evidence=e,rule=r,application=a,general=g,units=units,tags=[tag],
        note=note or '원본 PNG를 직접 보고 관찰 내용과 보기·정답을 대조. 작성 모델 자체 검토이며 C 교차 검수 대기.',
        rule_note='잠금 원본 lecture/solution 및 공통 내용표와 대조. 이미지 관찰과 배경 규칙을 구분.',
        source_disagreement=disagreement)

add('10126',
 'The May through October bars are all about 3 inches high. January is taller than March, and October is shorter than March.',
 'Bar height represents precipitation: similar heights mean similar amounts, and a shorter bar means a drier month.',
 'The similar May-through-October bars support option A. March is neither the wettest month nor drier than October.',
 [('ra','Option A fits the graph: similar bar heights indicate similar precipitation, so the May-through-October pattern supports roughly equal monthly amounts.'),
  ('e','Those six bars are about 3 inches high; January is taller than March, while October is shorter than March.'),
  ('ra','A shorter bar means a drier month, so March is neither the wettest month nor drier than October.')],
 'bar_chart',disagreement='원본 solution은 정답 A도 incorrect라고 표기한 뒤 A를 지지하는 설명을 붙임. 실제 막대와 원본 answer=0을 대조하여 모순 문구를 풀이에 복제하지 않음.')
add('10161',
 'The dark-green region occupies the western part of the northern Eurasian landmass, above the Mediterranean Sea and east of the Atlantic Ocean.',
 'Europe is the continent west of Asia and north of the Mediterranean; match the highlighted region to that geographic position.',
 'The highlighted western Eurasian region matches Europe, option B.',
 [('r','Europe lies west of Asia and north of the Mediterranean, which gives a geographic reference for identifying the highlighted continent.'),
  ('e','The dark-green area occupies western northern Eurasia, above the Mediterranean and east of the Atlantic.'),
  ('a','That position identifies Europe, so B is correct.')],'continent_map')
add('10168',
 'The outlined region contains blue shades. The legend places -10 degrees C in that blue range; 0 and 5 degrees C correspond to warmer, nonblue colors.',
 'Read temperature by matching colors inside the outline to the map legend, rather than using colors elsewhere on the map.',
 'Of the three choices, -10 degrees C matches a color inside the outline, so choose B.',
 [('ra','Choose B, -10 degrees C, by matching the colors inside the outline to the legend instead of colors elsewhere on the map.'),
  ('e','The outlined area has blue shades, and -10 degrees C falls in that blue range on the legend. The colors for 0 and 5 degrees C are warmer, nonblue shades.')],
 'color_scale',disagreement='원본 solution에 문맥 없는 5°C. 단편이 있음. 지도 범례·윤곽 내 색과 원본 answer=1은 -10°C를 지지함.')

def kinetic(sid,m1,m2,v1,v2,winner,letter):
    e=f'Sample A has particle mass {m1} u and average speed {v1:,} m/s; sample B has particle mass {m2} u and average speed {v2:,} m/s.'
    if m1==m2:
        r='In this particle model, at equal particle mass, the sample with the higher average speed has greater average kinetic energy and therefore higher temperature.'
        a=f'The masses are equal and sample {winner} has the higher average speed, so sample {winner} has the higher temperature, option {letter}.'
        parts=[('r','For equal-mass particles in this model, a higher average speed indicates greater average kinetic energy and a higher temperature.'),('e',e),('a',f'The equal masses let us compare speeds directly: sample {winner} is faster on average, so choose {letter}, sample {winner}.')]
    else:
        r='In this particle model, at the same average speed, the sample with heavier particles has greater average kinetic energy and therefore higher temperature.'
        a=f'The average speeds are equal, but sample {winner} has heavier particles, so its temperature is higher, option {letter}.'
        parts=[('ra',f'Sample {winner}, option {letter}, is hotter under the model: at the same average speed, heavier particles imply greater average kinetic energy and higher temperature.'),('e',e),('a',f'The shared speed makes the heavier particles in sample {winner} decisive.')]
    add(sid,e,r,a,parts,'particle_kinetic_energy',note='그림의 질량·평균속도 수치를 직접 확인. ScienceQA의 평균속도 비교 모형을 명시하고 평균속도의 제곱이 일반적으로 평균제곱속도와 같다고 주장하거나 정확한 온도비를 계산하지 않음.')
kinetic('10172',30,34,1600,1600,'B','C')
kinetic('10215',26,26,1900,1700,'A','B')
kinetic('10364',17,17,2000,1800,'A','C')

add('10177',
 'Both containers are labeled 50 mL of solvent. Solution A shows more yellow solute particles than solution B.',
 'Concentration compares solute particle count with solvent volume; at equal volumes, more solute particles mean higher concentration.',
 'With the same 50 mL volume and more yellow particles, solution A is more concentrated, option A.',
 [('r','At equal solvent volumes, the solution with more solute particles is more concentrated because concentration compares particle count with volume.'),('e','Both labels read 50 mL, and the drawing shows more yellow particles in solution A than in B.'),('a','Thus solution A has the higher concentration, giving option A.')],'concentration')
add('10186',
 'The table lists rock planets Mercury, Venus, Earth, and Mars with volumes 60, 930, 1,090, and 160 billion cubic kilometers, respectively. The other rows are gas or ice.',
 'To find the largest rocky planet, restrict the table to rock composition and compare the listed volumes.',
 'Earth has the greatest volume among the rock rows, so the statement is true, option B.',
 [('ra','The statement is true, option B, because Earth has the largest volume after restricting the comparison to planets listed as rock.'),('e','Those rows give Mercury 60, Venus 930, Earth 1,090, and Mars 160 billion cubic kilometers; the remaining planets are labeled gas or ice.'),('r','Volume measures size, so the largest number within the rock group determines the answer.')],'table_filter_compare')
add('10209',
 'The highlighted blue ocean lies west of the Americas and east of Asia and Australia, appearing at both outer edges of the world map.',
 'The Pacific Ocean lies between those landmasses and wraps across the edges of this map projection.',
 'The highlighted waters match the Pacific Ocean, option D.',
 [('r','The Pacific Ocean is between the Americas and Asia and Australia, so this map projection shows it across both outer edges.'),('e','The highlighted water occupies those edges, west of the Americas and east of Asia and Australia.'),('a','It therefore represents the Pacific Ocean, D.')],'ocean_map')
add('1022',
 'A small island in the eastern Caribbean island arc is boxed and highlighted, southeast of the large island containing the Dominican Republic.',
 'Dominica is a small island in the Lesser Antilles; the Dominican Republic occupies part of the much larger island of Hispaniola.',
 'The marked small island is Dominica, not the large Dominican Republic region, so choose B.',
 [('ra','Dominica, option B, fits the small highlighted island; the Dominican Republic instead occupies part of the much larger Hispaniola.'),('e','The box lies in the eastern Caribbean island arc, southeast of the large island containing the Dominican Republic.'),('r','This location matches Dominica in the Lesser Antilles.')],'country_map')

def direction(sid,direction,winner,others,letter):
    axis={'north':('N','up','above','uppermost'),'south':('S','down','below','lowermost'),'west':('W','left','to the left of','leftmost')}[direction]
    n,arrow,rel,extreme=axis
    e=f'The compass shows {n} pointing {arrow}. The mapped region of {winner} lies {rel} {others}.'
    r=f'Compare state regions along the indicated {direction} direction; the {extreme} region among these choices is farthest {direction}.'
    a=f'{winner} is the {extreme} of the compared regions, so it is farthest {direction}, option {letter}.'
    parts=[('r',f'To find the state farthest {direction}, compare the actual regions in the compass-indicated direction; here the {extreme} region wins.'),('e',f'The compass has {n} pointing {arrow}, and {winner}\'s region lies {rel} {others}.'),('a',f'This makes {winner} farthest {direction}, corresponding to {letter}.')]
    add(sid,e,r,a,parts,'state_direction',note='나침반과 주 경계를 직접 확인. 외부 라벨·연결선의 글자 위치 대신 실제 주 영역을 비교.')
direction('10233','north','New York','North Carolina, Kentucky, and Arkansas','A')
direction('10239','south','Texas','Maine, Nebraska, and Indiana','A')
direction('10328','north','Montana','West Virginia, Alabama, and New Jersey','D')
direction('1042','north','West Virginia','Tennessee, Florida, and Louisiana','A')
direction('10472','west','Florida','Rhode Island, Delaware, and Vermont','B')
direction('10518','north','Kentucky','Arkansas, Arizona, and Alabama','B')

add('10253',
 'The four labeled objects are wet paint, a wet bar of soap, butter, and yogurt.',
 'A slippery material is difficult to grip because surfaces slide over it easily. These wet or greasy materials share that property.',
 'Slippery applies to all four pictured materials, so the common property is option A.',
 [('ra','Slippery, option A, is the common property: slippery materials are difficult to grip because surfaces slide over them easily.'),('e','The pictures identify wet paint, a wet bar of soap, butter, and yogurt.'),('r','These wet or greasy materials all have that slippery quality.')],'object_property',note='이미지의 네 대상 및 라벨을 확인하고 일반적인 재료 성질을 적용. 촉감은 직접 실험한 관찰이라고 주장하지 않음.')
add('1026',
 'The equal-sized halves initially contain 3 yellow particles on the left and 5 on the right. The final diagram shows 4 on each side.',
 'Diffusion produces net movement from higher to lower concentration, although particles move both ways. Equal counts in equal volumes indicate equilibrium.',
 'The left side gains one particle overall while the right loses one, so more particles crossed to the left than to the right, option B.',
 [('r','Diffusion gives net movement from higher to lower concentration even though particles cross in both directions.'),('e','Here the equal-sized halves change from 3 left and 5 right to 4 on each side.'),('a','The left gains one particle overall and the right loses one, meaning more crossed leftward than rightward; B completes the sentence.'),('r','Equal counts in equal volumes show the final equilibrium.')],'diffusion',note='초기 좌3·우5, 최종 좌4·우4를 직접 계수. 좌 +1/우 -1의 순이동 확인. 한 입자만 움직였다고 주장하지 않음.')
add('10262',
 'The outline encloses a group of small Pacific islands far north of the Solomon Islands and northeast of Papua New Guinea. A detached dashed box appears farther north.',
 'Identify the country from its island-group location: the Marshall Islands lie in this northern Pacific region, east of the Federated States of Micronesia.',
 'The main highlighted island group matches the Marshall Islands, option B.',
 [('r','The Marshall Islands occupy this northern Pacific island region, east of the Federated States of Micronesia.'),('e','The main outline is northeast of Papua New Guinea and far north of the Solomon Islands, with a detached dashed box farther north.'),('a','Matching the main island group to that position gives the Marshall Islands, B.')],'country_map',note='본 군도 윤곽을 중심으로 식별. 점선 분쟁 섬 표시는 관찰만 기록하며 영유권 확정 주장을 추가하지 않음.')
add('10293',
 'Both gaps are 0.7 cm. Pair 1 has two small magnets; Pair 2 has one larger magnet and one small magnet. Pair 1 faces S to N, while Pair 2 faces N to N.',
 'In the stated same-material model, a larger magnet increases force magnitude at equal spacing. Pole reversal changes attraction to repulsion, not this size comparison.',
 'Pair 1 has the smaller magnet combination and therefore the smaller force magnitude, option C.',
 [('r','Under the same-material model, compare sizes at equal spacing; a larger magnet gives a greater force magnitude, while reversing poles changes attraction to repulsion.'),('e','Each gap is 0.7 cm. Pair 1 uses two small magnets facing S to N; Pair 2 uses one larger and one small magnet facing N to N.'),('a','Pair 1 consequently has the smaller force magnitude, so C is correct.')],'magnetic_force',note='그림 간격 0.7 cm 및 큰 자석이 Pair 2 왼쪽임을 확인. 원본 solution의 존재하지 않는 A/B 라벨을 복제하지 않음. 동일 재료·동일 형상 계열의 교육용 크기 모형 적용.')
add('10296',
 'Both solutions have 50 mL of solvent, and solution A contains more green particles than solution B.',
 'Solute concentration is particle count per solvent volume, so at equal volumes the larger count gives the higher concentration.',
 'Solution A has more green particles per equal volume, so it has the higher concentration, option B.',
 [('ra','The higher concentration is in solution A, which is option B, because more particles in the same volume mean a larger particle-count-to-volume ratio.'),('e','Both solvent labels read 50 mL, while solution A visibly has more green particles than B.')],'concentration')
add('10302',
 'Both pairs show equally sized magnets with 1.4 cm gaps, and each pair has an N pole facing an S pole.',
 'In this same-material comparison, matching magnet sizes, spacing, and facing-pole arrangement give matching magnetic force magnitudes.',
 'The relevant conditions match, so the force magnitude is the same in both pairs, option A.',
 [('r','Equal sizes, spacing, and pole arrangements give equal force magnitudes for these same-material magnets.'),('e','The drawing shows matching sizes and 1.4 cm gaps in both pairs, with N facing S each time.'),('a','Since the conditions match, select A: the force magnitude is the same in both pairs.')],'magnetic_force')
add('10310',
 'The arrows show green algae to water flea, water flea to rotifer, and rotifer to black crappie.',
 'Arrows point from food to eater. A producer is eaten by a primary consumer, which is eaten by a secondary consumer; eating that secondary consumer makes a tertiary consumer on that path.',
 'Starting with green algae as the producer, this path makes water flea primary, rotifer secondary, and black crappie tertiary, so choose B.',
 [('r','Follow arrows from food to eater, counting from a producer through primary and secondary consumers to a tertiary consumer.'),('e','One visible path is green algae to water flea to rotifer to black crappie.'),('a','With green algae as producer, water flea is primary, rotifer secondary, and black crappie tertiary on this path. Thus B is correct.')],'food_web',note='교차 화살표의 머리를 따라 green algae→water flea→rotifer→black crappie 경로 확인. 복수 영양단계 가능성을 배제하지 않고 해당 경로로 한정.')
add('10315',
 'The horizontal line marked 0 degrees latitude crosses Africa. Australia is below it, and North America is above it.',
 'The equator is the line of 0 degrees latitude; a continent is intersected when that line passes through its land area.',
 'Of the listed continents, the line crosses Africa, so choose B.',
 [('ra','Africa, option B, is intersected by the equator because the equator is 0 degrees latitude and must pass through the continent\'s land.'),('e','On the map, that horizontal 0-degree line crosses Africa, with Australia below it and North America above it.')],'coordinate_map')
add('10331',
 'The marker is on the Pacific coast of California near a bay, well north of the southern end of the state.',
 'San Francisco is on the northern California coast; San Diego is much farther south, Portland is in Oregon, and Salt Lake City is inland.',
 'The marked coastal California location matches San Francisco, option C.',
 [('ra','San Francisco, C, fits this coastal California marker.'),('e','The point sits near a bay on the Pacific coast, well north of California\'s southern end.'),('r','San Francisco is on that northern California coast; San Diego is much farther south, Portland is in Oregon, and Salt Lake City is inland.')],'city_map')
add('10341',
 'The box marks a small island near the southern end of the eastern Caribbean island arc, northwest of Trinidad and Tobago.',
 'Grenada lies at the southern end of the Lesser Antilles, south of Saint Vincent and the Grenadines and northwest of Trinidad and Tobago.',
 'That southern island location corresponds to Grenada, option B.',
 [('r','Grenada lies south of Saint Vincent and the Grenadines and northwest of Trinidad and Tobago, near the southern end of the Lesser Antilles.'),('e','The highlighted box occupies that southern part of the eastern Caribbean arc, northwest of Trinidad and Tobago.'),('a','The location identifies Grenada, B.')],'country_map')
add('10378',
 'The large landmass south of Papua New Guinea is dark green, together with the smaller island just off its southeastern coast.',
 'Australia includes the large Australian mainland and Tasmania, the island just south of its southeastern corner.',
 'The highlighted mainland and Tasmania identify Australia, option C.',
 [('ra','Australia, option C, matches the highlighted combination of mainland and Tasmania.'),('e','The dark-green mainland lies south of Papua New Guinea, and a small island off its southeastern coast is also shaded.'),('r','Tasmania is part of Australia and lies just south of that mainland corner.')],'country_map')
add('1038',
 'The dark-green areas include a coastal band in New England and a separate large region farther northeast.',
 'Colonial Massachusetts included present-day Massachusetts and land that later became Maine; historical colony borders differ from modern state borders.',
 'The two highlighted regions match colonial Massachusetts and its Maine territory, so choose C.',
 [('r','Colonial Massachusetts included present-day Massachusetts and the territory that later became Maine, so modern state borders alone are insufficient.'),('e','The map shades a New England coastal band and a separate large region farther northeast.'),('a','Together they identify Massachusetts and its Maine territory, making C correct.')],'historical_map',note='분리된 두 강조 영역 확인. Maine 정부 역사 페이지 https://www.maine.gov/portal/about_me/history.html 에서 Maine의 Massachusetts 소속 및 1820 분리 확인. 원본 solution의 절대적 역사 주장(Maine was never its own colony)은 추가하지 않음.')
add('10393',
 'The picture shows a blue water slide with water running along its surface.',
 'Slippery describes a surface on which objects slide easily; yellow describes a color, which is not the blue color shown.',
 'The wet slide is better described as slippery than yellow, so choose B.',
 [('ra','Slippery, B, is the better description because a slippery surface lets objects slide easily.'),('e','The pictured water slide is blue and has water running along it.'),('r','Yellow is a color description and does not match the blue shown.')],'object_property')
add('10405',
 'The corresponding magnets have matching sizes. Pair 1 is separated by 21.4 mm and Pair 2 by 45.8 mm; opposite poles face across each gap.',
 'With otherwise matching magnets, a greater separation gives a smaller magnetic force magnitude.',
 'Pair 2 has the larger gap, 45.8 mm, so its force magnitude is smaller, option B.',
 [('r','Greater separation weakens the magnetic force when the compared magnets otherwise match.'),('e','The corresponding magnets have matching sizes and opposite facing poles, but the gaps are 21.4 mm in Pair 1 and 45.8 mm in Pair 2.'),('a','Since 45.8 mm is larger, Pair 2 has the smaller force magnitude; choose B.')],'magnetic_force')
add('10448',
 'Several separate toe bones project from the feet beneath the fossil\'s large shell. No long flippers are depicted.',
 'A trait observable on a fossil must be supported by the preserved body structures shown.',
 'The visible foot digits support toes, option B.',
 [('ra','Toes, B, are supported by the fossil because an observable trait needs visible preserved body structures.'),('e','The feet below the large shell have several separate projecting toe bones, while no long flippers are depicted.')],'fossil_trait',note='다리 끝의 분리된 발가락 구조를 확대 없이 원본에서 확인. hint의 연대 주장은 풀이에 사용·재검증하지 않음; 원문은 보존.')
add('10477',
 'The box encloses a short diagonal chain of small Pacific islands northeast of Australia and north of Fiji.',
 'Tuvalu is the small island chain north of Fiji; the Federated States of Micronesia is much farther northwest, and Tonga lies farther south.',
 'The boxed chain matches Tuvalu, option C.',
 [('ra','Tuvalu, option C, fits the boxed island chain north of Fiji.'),('e','The map shows a short diagonal group northeast of Australia and north of Fiji.'),('r','Tuvalu occupies this location, while the Federated States of Micronesia lies much farther northwest and Tonga farther south.')],'country_map')
add('10484',
 'The column headers are f and f, the row headers are F and F, and all four offspring boxes contain Ff.',
 'Heterozygous means having two different alleles. Each Punnett-square box is an equally likely outcome, so probability is favorable boxes divided by total boxes.',
 'All four Ff outcomes are heterozygous, giving 4/4, option D.',
 [('r','A heterozygous offspring has two different alleles, and its probability is the fraction of equally likely Punnett-square boxes with that genotype.'),('e','The f, f columns and F, F rows produce Ff in all four boxes.'),('a','Every box is heterozygous, so the probability is 4/4, which is D.')],'punnett_square',note='상단 ff·좌측 FF 및 내부 4칸 Ff를 직접 확인. 4/4=1. 원본 solution은 빈 문자열이며 lecture 정의와 표로 자체 풀이 작성. 원본 5지 보기 유지.')
add('10488',
 'The pictured jeans are colored blue.',
 'Blue is a visible color property; hard describes resistance to changing shape and is not the usual property of flexible denim clothing.',
 'The visible blue color gives the better matching property, option A.',
 [('ra','Blue, option A, matches the jeans because blue names a visible color property.'),('e','The illustration colors the jeans blue.'),('r','Hard instead describes resistance to changing shape, which is not the usual property of flexible denim clothing.')],'object_property')
add('10495',
 'The skeleton has large curved horns extending from its skull. The pictured remains do not show a striped coat.',
 'Visible preserved structures can support a fossil trait; a coat pattern cannot be established from bare bones in this image.',
 'The observed head structures support horns on its head, option B.',
 [('ra','Horns on its head, B, is supported by visible preserved structures.'),('e','Large curved horns extend from the skull, while no striped coat is shown.'),('r','Bare bones in this picture cannot establish a coat pattern, so that alternative is not observable here.')],'fossil_trait')
add('10541',
 'The dark-green region covers the eastern part of the Eurasian landmass, extending from the far north down toward the Indian Ocean and Southeast Asia.',
 'Asia occupies eastern Eurasia; the continent can be identified by this broad landmass north of the Indian Ocean.',
 'The highlighted eastern Eurasian region is Asia, option A.',
 [('r','Asia occupies eastern Eurasia, including the broad landmass north of the Indian Ocean.'),('e','The shaded region covers that eastern landmass from the far north to the Indian Ocean and Southeast Asia.'),('a','Its extent identifies Asia, so choose A.')],'continent_map')
add('1055',
 'A long stepped outline encloses widely separated island groups across the central Pacific, northeast of Australia.',
 'Kiribati is spread across widely separated central Pacific island groups, whereas Palau, Tuvalu, and Nauru occupy much smaller geographic spans.',
 'The broad highlighted span matches Kiribati, option D.',
 [('ra','Kiribati, D, matches the unusually wide highlighted span.'),('e','The stepped outline runs across separated central Pacific island groups northeast of Australia.'),('r','Kiribati extends across widely separated groups; Palau, Tuvalu, and Nauru have much smaller geographic spans.')],'country_map')
add('10553',
 'Both pairs have matching magnet sizes and 5 in gaps. Pair 1 faces N to S, and Pair 2 faces S to N.',
 'Both arrangements are opposite-pole pairs. For matching same-material magnets at equal spacing, reversing both orientations preserves the force strength.',
 'The matching sizes, gaps, and opposite-pole arrangements give the same strength in both pairs, option A.',
 [('r','Reversing both magnets preserves the force strength when their sizes, material, and spacing match and opposite poles still face.'),('e','Both gaps are 5 in and the sizes match: Pair 1 faces N to S, while Pair 2 faces S to N.'),('a','Both remain opposite-pole arrangements, so the force strength is the same, A.')],'magnetic_force')
add('1058',
 'The highlighted island lies directly south of Cuba and west of the large island containing Haiti.',
 'Jamaica is the Caribbean island south of Cuba and west of Hispaniola, the island shared by Haiti and the Dominican Republic.',
 'That relative position identifies Jamaica, option B.',
 [('r','Jamaica lies south of Cuba and west of Hispaniola, which is shared by Haiti and the Dominican Republic.'),('e','The dark-green island is directly below Cuba and west of the large island containing Haiti.'),('a','This location matches Jamaica, making B correct.')],'country_map')
add('10588',
 'The pictured animal has arms that split repeatedly into many fine branches around its central body.',
 'The passage distinguishes basket stars by repeatedly branched arms, unlike the thin snake-like arms of brittle stars.',
 'The repeated branching matches basket star, option A.',
 [('r','Repeatedly branched arms distinguish basket stars from the thin snake-like arms described for brittle stars.'),('e','The animal in the picture has arms splitting again and again into many fine branches around its center.'),('a','That branching matches a basket star, so A is correct.')],'visual_classification')
add('10590',
 'A small fin projects from the fish fossil\'s back. The fossil image does not show reddish-orange body scales.',
 'Choose a trait supported by preserved structures in the image; original living color cannot be inferred from the fossil\'s rock color.',
 'The visible back fin supports a small fin on its back, option A.',
 [('ra','A small fin on its back, A, is the observable trait because the answer must be supported by preserved structures.'),('e','The image shows a small fin projecting from the fish\'s back, without visible reddish-orange body scales.'),('r','The surrounding rock color does not establish the animal\'s original living color.')],'fossil_trait')
add('10593',
 'Corresponding magnets have matching sizes. Pair 1 has a 3.75 in gap with N facing N; Pair 2 has a 1.75 in gap with S facing S.',
 'Both are like-pole arrangements. With matching same-material magnets, a smaller gap gives a greater magnetic force magnitude.',
 'Pair 2 has the smaller gap, so the magnitude is greater in Pair 2, option C.',
 [('ra','Pair 2 has the greater force magnitude, C, because decreasing the gap strengthens the force between otherwise matching magnets.'),('e','The sizes match, but Pair 1 has a 3.75 in gap with N facing N and Pair 2 a 1.75 in gap with S facing S.'),('r','Both are like-pole arrangements, so the smaller gap is the deciding difference.')],'magnetic_force')
add('10606',
 'The pictures show an apron, a ball of wet clay, stuffed dice, and a stuffed puppy. The apron and clay are not fuzzy; the dice have a fuzzy surface.',
 'Soft objects yield when pressed or squeezed. These fabric, wet-clay, and stuffed objects share softness; fuzzy and smooth do not describe all four.',
 'Soft is the property common to all four, option C.',
 [('ra','Soft, option C, applies to all four because soft objects yield when pressed or squeezed.'),('e','The objects are an apron, wet clay, stuffed dice, and a stuffed puppy; the apron and clay are not fuzzy, while the dice have a fuzzy surface.'),('r','Fabric, wet clay, and stuffing share softness, whereas neither fuzzy nor smooth describes the whole group.')],'object_property',note='그림의 대상·표면 묘사를 확인. 부드러움은 식별한 재료의 일반 성질로 적용했으며 직접 누르는 실험을 했다고 주장하지 않음.')
add('10612',
 'The upper ring magnet has S above N, and the lower ring has N above S. The two N poles face each other across the gap.',
 'Like magnetic poles repel, while opposite poles attract; the facing poles determine this interaction.',
 'Because N faces N, the magnets repel, option A.',
 [('r','Like poles repel and opposite poles attract, so inspect the poles facing across the gap.'),('e','The upper ring is S above N and the lower is N above S, placing N opposite N.'),('a','Those like facing poles repel, so A is correct.')],'magnetic_poles')
add('10617',
 'The pictures identify a metal nail, a water pitcher, a metal bar, and a car bumper. The pitcher is shown as see-through.',
 'Hard objects resist being pressed into a different shape. These rigid objects share hardness; opaque cannot describe all four because the pitcher is transparent.',
 'Hard is the shared property, option C.',
 [('ra','Hard, C, is shared by these rigid objects: hardness means resisting being pressed into a different shape.'),('e','They are a metal nail, a water pitcher, a metal bar, and a car bumper, with the pitcher depicted as see-through.'),('r','That transparent pitcher also prevents opaque from being a property of all four.')],'object_property')

def shorten_era(sid,**changes):
    c=DATA[sid]
    for key,new in changes.items():
        old=c[key];c[key]=new
        for u in c['units']:
            if u[0]==old:u[0]=new
            if u[2]==old:u[2]=new
    c['note']+=' 길이 검토 후 ERA의 반복 표현을 축약; 의미·수치·조건 유지, 일반 풀이 길이 늘리기 없음.'
shorten_era('10161',evidence='The dark-green area occupies western Eurasia, north of the Mediterranean and east of the Atlantic.')
shorten_era('10168',rule='Match colors inside the outline to the temperature legend, rather than colors elsewhere.')
shorten_era('10177',application='Thus solution A has the higher concentration, option A.')
shorten_era('1022',evidence='The boxed small island lies in the eastern Caribbean arc, southeast of the island containing the Dominican Republic.',rule='Dominica is in the Lesser Antilles; the Dominican Republic occupies part of the much larger Hispaniola.',application='The highlighted small island matches Dominica, option B.')
shorten_era('10262',evidence='The outline surrounds small Pacific islands north of the Solomon Islands and northeast of Papua New Guinea; a detached dashed box lies farther north.',rule='The Marshall Islands occupy this northern Pacific region, east of the Federated States of Micronesia.',application='This main island group matches the Marshall Islands, B.')
shorten_era('10293',application='The smaller magnet combination in Pair 1 gives the smaller force magnitude, option C.')
shorten_era('10296',rule='Concentration is solute count per solvent volume; at equal volumes, more particles mean higher concentration.',application='Thus solution A has the higher concentration, option B.')
shorten_era('10310',rule='Arrows point from food to eater. Along a feeding path, levels proceed from producer to primary, secondary, then tertiary consumer.',application='Here green algae is producer, water flea primary, rotifer secondary, and black crappie tertiary: option B.')
shorten_era('10315',application='Thus the equator intersects Africa, option B.')
shorten_era('10393',application='Thus slippery is the better answer, option B.')
shorten_era('10448',application='These visible digits support toes, option B.')
shorten_era('10541',evidence='The dark-green area covers eastern Eurasia from the far north to the Indian Ocean and Southeast Asia.',rule='Asia occupies eastern Eurasia, including the broad landmass north of the Indian Ocean.',application='This highlighted extent identifies Asia, option A.')
shorten_era('1055',rule='Kiribati spans widely separated central Pacific island groups; Palau, Tuvalu, and Nauru cover much smaller areas.',application='This broad highlighted span identifies Kiribati, D.')
shorten_era('10612',application='The facing N poles therefore repel, option A.',rule='Like poles repel and opposite poles attract; inspect the facing poles.')
assert len(DATA)==45,len(DATA)
queue=[json.loads(x) for x in (ROOT/'candidate_queue.jsonl').read_text(encoding='utf-8').splitlines()]
DATA={q['source_id']:DATA[q['source_id']] for q in queue if q['source_id'] in DATA}
(ROOT/'author_content_new.json').write_text(json.dumps(DATA,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Authored',len(DATA),'paired common-content records; last ID',list(DATA)[-1])
