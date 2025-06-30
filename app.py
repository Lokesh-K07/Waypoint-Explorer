import heapq
from typing import Dict, List, Tuple
import folium
import osmnx as ox
import networkx as nx
import os
import pickle
from functools import lru_cache
from flask import Flask, render_template, request, url_for, jsonify, session
from math import sqrt
import logging
from flask_session import Session 
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import warnings 

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['SESSION_TYPE'] = 'filesystem' 
Session(app)

MAPS_DIR = "static/maps"
if not os.path.exists(MAPS_DIR):
    os.makedirs(MAPS_DIR)

visakhapatnam_data = {
     "general_locations": {
    "1": {"name": "Tagarapuvalasa", "coords": (17.933198096970433, 83.42585870989144)},
    "2": {"name": "Anil Neerukonda", "coords": (17.92200575653113, 83.42549895767064)},
    "3": {"name": "NRI College Road", "coords": (17.921056683330576, 83.42593067669937)},
    "4": {"name": "Sanghavalasa", "coords": (18.805308738783488, 83.34986920341255)},
    "5": {"name": "Samyyavalasa Road", "coords": (17.934423165519423, 83.39421151534125)},
    "6": {"name": "Valandapeta Village", "coords": (17.916704614236938, 83.41431996106505)},
    "7": {"name": "R Tallavalasa", "coords": (17.909275704081185, 83.41180470046785)},
    "8": {"name": "Peddipalem Road", "coords": (17.90268756543267, 83.40007481534126)},
    "9": {"name": "Vemulavalasa", "coords": (17.895949589532126, 83.38166015334748)},
    "10": {"name": "Dukkavanipalem", "coords": (17.884692363279406, 83.35641082764236)},
    "11": {"name": "Gudilova", "coords": (17.883301168618196, 83.32493752157218)},
    "12": {"name": "Sontyam", "coords": (17.86770462063886, 83.29915322501525)},
    "13": {"name": "Dibbameedapalem", "coords": (17.85248793821026, 83.2964666277239)},
    "14": {"name": "Mamidilova", "coords": (17.84423650185978, 83.29003429427173)},
    "15": {"name": "Mindivanipalem", "coords": (17.87927822590642, 83.31001282742614)},
    "16": {"name": "Neelakondilu", "coords": (17.876450862915554, 83.30444306688072)},
    "17": {"name": "Gandigudam", "coords": (17.843379814787554, 83.25614322730046)},
    "18": {"name": "SR Puram", "coords": (17.827789671590963, 83.24872087187111)},
    "19": {"name": "Gurrampalem", "coords": (17.820053810342973, 83.19751820314637)},
    "20": {"name": "Pulaganipalem", "coords": (17.81406198347314, 83.22245086658512)},
    "21": {"name": "Dabbanda", "coords": (17.818400094345947, 83.27429467719128)},
    "22": {"name": "Bhairava Kona", "coords": (17.80722276649315, 83.25565569578558)},
    "23": {"name": "Pendurti", "coords": (17.805221040665053, 83.21020440112734)},
    "24": {"name": "Chinnamushidivada", "coords": (17.77793964966878, 83.18498895788419)},
    "25": {"name": "Chintagatla Agraharam", "coords": (17.781290842017526, 83.16208381549131)},
    "26": {"name": "Lakshmi puram", "coords": (17.76709044342686, 83.16804869632278)},
    "27": {"name": "Vepagunta", "coords": (17.776072399414062, 83.21859546029869)},
    "28": {"name": "Pulagalipalem", "coords": (17.789211436326745, 83.1934319829372)},
    "29": {"name": "Valimeraka", "coords": (17.78953834346261, 83.21068395012202)},
    "30": {"name": "Peda narava", "coords": (17.74383859926407, 83.18756842560941)},
    "31": {"name": "Duvvada", "coords": (17.697166029111422, 83.14637930708187)},
    "32": {"name": "Lakkarajupalem", "coords": (17.69502850451003, 83.15269350118841)},
    "33": {"name": "Marripalem", "coords": (17.670713156035102, 83.08557507276053)},
    "34": {"name": "Andhra Kesri Nagar", "coords": (17.745197111503344, 83.24919640953192)},
    "35": {"name": "Lankelapalem", "coords": (17.662634096232573, 83.08413326639457)},
    "36": {"name": "Edulapaka Bonangi", "coords": (17.61448412831922, 83.0778499)},
    "37": {"name": "Ravada Road", "coords": (17.601388112803146, 83.07384163122536)},
    "38": {"name": "Wada cheepurupalli Road", "coords": (17.56609860369058, 83.07387784107911)},
    "39": {"name": "Tantadi Road", "coords": (17.609008647604334, 83.0769261716208)},
    "40": {"name": "Mutyalammapalem Road", "coords": (17.536849952998057, 83.08818712273003)},
    "41": {"name": "Appikonda Bridge", "coords": (17.55502877502633, 83.1404278018472)},
    "42": {"name": "Palavalasa Beach", "coords": (17.556335474583502, 83.14569937116468)},
    "43": {"name": "Appikonda Road", "coords": (17.61498566094003, 83.16284354232936)},
    "44": {"name": "Appikonda Beach", "coords": (17.567844917481366, 83.17179325341256)},
    "45": {"name": "Devara-Konda", "coords": (17.59263732455765, 83.21115347116469)},
    "46": {"name": "JoJo’s Land", "coords": (17.59628211471208, 83.21497178835313)},
    "47": {"name": "Dibbapalem", "coords": (17.632415681029, 83.23650119481013)},
    "48": {"name": "Gangavaram", "coords": (17.63894994337121, 83.23718014830895)},
    "49": {"name": "Yarada Beach View Point", "coords": (17.646473939944013, 83.2472669288353)},
    "50": {"name": "Yarada Sea mountain view point", "coords": (17.646019284351212, 83.25979849939155)},
    "51": {"name": "Roki Yarada", "coords": (17.648491200000006, 83.26074368650595)},
    "52": {"name": "Vizag Yarada Beach", "coords": (17.655575956234788, 83.26787087836678)},
    "53": {"name": "Four road junction (Yarada)", "coords": (17.658546579713505, 83.2743092711647)},
    "54": {"name": "Kothuru Colony", "coords": (17.660107100000012, 83.2767350288353)},
    "55": {"name": "Beach View Point(Yarada)", "coords": (17.646381923997023, 83.24728838650593)},
    "56": {"name": "Dolphin cove(NAVY)", "coords": (17.666864645667932, 83.28246419999998)},
    "57": {"name": "Jolly Green Paradise Beach", "coords": (17.67089031688659, 83.28970044835943)},
    "58": {"name": "Dolphin’s Nose", "coords": (17.676479222287163, 83.29261988650595)},
    "59": {"name": "Durga Beach", "coords": (17.68186135387982, 83.29545375)},
    "60": {"name": "Pipeline View Point", "coords": (17.6837103, 83.29636839999999)},
    "61": {"name": "Visakhapatnam Dock View point", "coords": (17.687206131590337, 83.29335026428166)},
    "62": {"name": "Dolphin hill road", "coords": (17.67417174308313, 83.2748503981455)},
    "63": {"name": "HSL Drydock", "coords": (17.68827366485073, 83.2791734711647)},
    "64": {"name": "Scindia Road", "coords": (17.694906722427888, 83.26324582084627)},
    "65": {"name": "Pipeline Jetty", "coords": (17.6950868716716, 83.27423349007728)},
    "66": {"name": "ND vizag", "coords": (17.70601936182035, 83.26320377116467)},
    "67": {"name": "MG bridge", "coords": (17.71170682028226, 83.2634289)},
    "68": {"name": "Toll Gate Port Road", "coords": (17.71514360043805, 83.26785364417657)},
    "69": {"name": "West Ore Berth", "coords": (17.72007983411942, 83.2856527819624)},
    "70": {"name": "Vizag SeaPort", "coords": (17.710968059353085, 83.28392665582342)},
    "71": {"name": "Port Blair- Vizag Ships(Port)", "coords": (17.703141019108184, 83.28474372609993)},
    "72": {"name": "Vizag Docks Main Gate", "coords": (17.697410800000004, 83.28398305767064)},
    "73": {"name": "Marine Foremen Complex", "coords": (17.6929061854144, 83.28434512883531)},
    "74": {"name": "Ambusanga St", "coords": (17.691327485773762, 83.2859848711647)},
    "75": {"name": "Ross Hill Church", "coords": (17.690439346694937, 83.28706383690155)},
    "76": {"name": "Vizag Port Trust Drydock", "coords": (17.689399043105176, 83.28414138650595)},
    "77": {"name": "Sri Venkateswara Swamy Temple", "coords": (17.68955656339816, 83.28998341298603)},
    "78": {"name": "Beach Road", "coords": (17.790031810687204, 83.3847151288353)},
    "79": {"name": "1 town water tank", "coords": (17.69748325000001, 83.2994996306825)},
    "80": {"name": "Port Area", "coords": (17.697757137894516, 83.2968622789915)},
    "81": {"name": "Kanchara Veedi", "coords": (17.6988707, 83.29959638650593)},
    "82": {"name": "Golla Veedi", "coords": (17.70093152615197, 83.30332235795245)},
    "83": {"name": "Adarsanagar", "coords": (17.92672821975557, 83.42565283448151)},
    "84": {"name": "Fishery Survey Of India", "coords": (17.70136827638447, 83.30471739389473)},
    "85": {"name": "Rock Beach View Point", "coords": (17.70280530000001, 83.30792742883531)},
    "86": {"name": "Naval Quarters", "coords": (17.70445693840522, 83.30861538577321)},
    "87": {"name": "Chakram", "coords": (17.706384179415174, 83.31130567116466)},
    "88": {"name": "Hawa Mahal", "coords": (26.924026668175756, 75.8267036711647)},
    "89": {"name": "Varun Inox Road", "coords": (17.711012100000012, 83.3157591288353)},
    "90": {"name": "Indira Gandhi Children Park", "coords": (17.710527007099753, 83.31726312273004)},
    "91": {"name": "RK Beach", "coords": (17.71244758569796, 83.32062174767657)},
    "92": {"name": "Pandurangapuram", "coords": (17.715498714932362, 83.32193662131647)},
    "93": {"name": "Harbour Park Road", "coords": (17.714851500521256, 83.31797979999999)},
    "94": {"name": "Naval Canteen", "coords": (17.7023571928511, 83.30574965018063)},
    "95": {"name": "Natural View Point", "coords": (17.7030922, 83.30847729999999)},
    "96": {"name": "Bellam Vinayakudu Temple", "coords": (17.70439222069886, 83.30750052883532)},
    "97": {"name": "Coastal battery beach junction", "coords": (17.7052708, 83.3098921288353)},
    "98": {"name": "Gokul Park", "coords": (17.707207519289547, 83.31275163281926)},
    "99": {"name": "Collector Office junction", "coords": (17.708677197734616, 83.30950807003732)},
    "100": {"name": "Novotel Visakhapatnam Varun Beach", "coords": (17.71087989154895, 83.31614769980074)},
    "101": {"name": "Harbour Park", "coords": (17.714821062728703, 83.31799037739248)},
    "102": {"name": "Submarine Museum", "coords": (17.716960525266817, 83.32928831883918)},
    "103": {"name": "VMRDA INS Kursura Submarine Museum", "coords": (17.71731235882686, 83.33014999883059)},
    "104": {"name": "Chinna Waltair Main Rd", "coords": (17.722205910123325, 83.32840769458406)},
    "105": {"name": "Victory At Sea War Memorial", "coords": (17.71871937953422, 83.3322483288353)},
    "106": {"name": "Visakha Museum", "coords": (17.720745141883143, 83.33404539020033)},
    "107": {"name": "Visakhapatnam Lighthouse", "coords": (17.720699439540248, 83.33796715767063)},
    "108": {"name": "VMRDA Park", "coords": (17.72223239841843, 83.33803153068249)},
    "109": {"name": "Vuda Park Skating Rink", "coords": (17.72435807970732, 83.3396623136502)},
    "110": {"name": "Karakachettu Rd", "coords": (17.730898479416958, 83.33833193807126)},
    "111": {"name": "Waltair Main Rd", "coords": (17.730653218735235, 83.33004927720891)},
    "112": {"name": "Lawsons Bay Beach Park", "coords": (17.733767961952314, 83.34233754232939)},
    "113": {"name": "MVP Colony", "coords": (17.741373669645736, 83.33433201332413)},
    "114": {"name": "MVP Main Rd", "coords": (17.743539981533374, 83.33658492883532)},
    "115": {"name": "HB Colony", "coords": (17.746493854492062, 83.32542925169899)},
    "116": {"name": "Sector 1", "coords": (17.743881072121834, 83.32977903358626)},
    "117": {"name": "Shivaji Park", "coords": (17.73756779023404, 83.33090750926849)},
    "118": {"name": "Ropeway Base Station", "coords": (17.745229318370793, 83.34577438650595)},
    "119": {"name": "Kailash Giri", "coords": (17.747712365185247, 83.34612843807133)},
    "120": {"name": "Tenneti Park", "coords": (17.749469891908603, 83.34775922103904)},
    "121": {"name": "Jodugullapalem", "coords": (17.750433532852952, 83.34858366298556)},
    "122": {"name": "Police Quarters Rd", "coords": (17.754253947632744, 83.34113029443431)},
    "123": {"name": "Visalakshi Nagar", "coords": (17.75376841627137, 83.34182911852632)},
    "124": {"name": "Walk in to beach", "coords": (17.755692420991707, 83.35299965025574)},
    "125": {"name": "Vizag View Point", "coords": (17.756489405525265, 83.35373993991995)},
    "126": {"name": "Dolphin Pool", "coords": (17.759103862122046, 83.35417457047852)},
    "127": {"name": "Vizag Zoo Park", "coords": (17.760686418617677, 83.35495785046497)},
    "128": {"name": "Sagar Nagar Beach", "coords": (17.764877367369063, 83.36501579399443)},
    "129": {"name": "ISKCON Temple", "coords": (17.76824889543836, 83.36668281176121)},
    "130": {"name": "Gayatri Vidya Parishad College for Degree and P.G. Courses (A)", "coords": (17.777661274196078, 83.37619599634708)},
    "131": {"name": "Sri Venkateswara Swamy Devalayam, TTD", "coords": (17.77896429451991, 83.37756979266976)},
    "132": {"name": "Rushikonda Beach", "coords": (17.782767557671182, 83.38518213930666)},
    "133": {"name": "Yendada Rushikonda Road", "coords": (17.784030313733382, 83.3806744942204)},
    "134": {"name": "Sai Priya Beach Resort", "coords": (17.785786899469773, 83.38450936771594)},
    "135": {"name": "A1 Grand The Convention", "coords": (17.787230727609987, 83.38448079088984)},
    "136": {"name": "Rushikonda", "coords": (17.7925841313019, 83.38408062756761)},
    "137": {"name": "IT Sez Beach", "coords": (17.79445068596565, 83.39211946981138)},
    "138": {"name": "Pedda Rushikonda", "coords": (17.797318105185592, 83.38686623592311)},
    "139": {"name": "IT Park Beach", "coords": (17.80061344437226, 83.39795893116901)},
    "140": {"name": "Ramanaidu Studios Road", "coords": (17.803456573716062, 83.39884043898233)},
    "141": {"name": "Rama Naidu Studios", "coords": (17.80970172509259, 83.3975282372409)},
    "142": {"name": "Vizag Film Nagar Cultural Centre", "coords": (17.812989506249192, 83.40819779984074)},
    "143": {"name": "100 Feet Road", "coords": (17.8141256464076, 83.40858624652873)},
    "144": {"name": "Thimmapuram", "coords": (17.814760301934044, 83.40844058319686)},
    "145": {"name": "Totlakonda Beach Park", "coords": (17.82228032070442, 83.41480058790098)},
    "146": {"name": "Totlakonda Mountain Entrance", "coords": (17.82262650476703, 83.41467015448316)},
    "147": {"name": "Sila Thoranam - Natural Arch", "coords": (17.824217664771364, 83.4158335705709)},
    "148": {"name": "Mangamaripeta Beach", "coords": (17.82523132195069, 83.41630890697692)},
    "149": {"name": "Lord Hanuman Statue", "coords": (17.827912880431732, 83.41479301115193)},
    "150": {"name": "Thotlakonda Buddhist Monastery", "coords": (17.8289714816475, 83.40914566418519)},
    "151": {"name": "Beach View Point", "coords": (17.831164039276477, 83.41141291099397)},
    "152": {"name": "Vizag Surf Club", "coords": (17.835107671209883, 83.41584244233984)},
    "153": {"name": "Kapuluppada", "coords": (17.835314749717774, 83.38451603817747)},
    "154": {"name": "Uppada Beach Road", "coords": (17.8434810397366, 83.40655804602986)},
    "155": {"name": "Uppada Beach", "coords": (17.85025991691982, 83.4135315847517)},
    "156": {"name": "INS Kalinga Beach", "coords": (17.855276373042532, 83.41698202637336)},
    "157": {"name": "Nerellavalasa Rural", "coords": (17.872349693076423, 83.42653588477266)},
    "158": {"name": "Kothavalasa", "coords": (17.88044153920261, 83.40799647726486)},
    "159": {"name": "Vellanki", "coords": (17.88147640719119, 83.39248130934543)},
    "160": {"name": "Boddapalem", "coords": (17.882970031509167, 83.38578003021215)},
    "161": {"name": "SOS Junction Beach", "coords": (17.878052938913854, 83.44524410218735)},
    "162": {"name": "Bheemili Beach", "coords": (17.893036601040286, 83.45564336798684)},
    "163": {"name": "Gollalapalem", "coords": (17.89254361244154, 83.42709060299792)},
    "164": {"name": "Bheemili Beach Road", "coords": (17.888472646504425, 83.4548257422222)},
    "165": {"name": "Sri Lakshmi Narasimha Swami Temple", "coords": (17.889115813403393, 83.44908588880955)},
    "166": {"name": "Buddhist Site, Pavurallakonda", "coords": (17.889205294320714, 83.4379194297037)},
    "167": {"name": "Nehru St", "coords": (17.89060622189773, 83.45211086875373)},
    "168": {"name": "Ramadaasu St", "coords": (17.891039155928734, 83.45021686231543)},
    "169": {"name": "Patel St", "coords": (17.89190502082264, 83.45183233839518)},
    "170": {"name": "Subhash St", "coords": (17.89296820300903, 83.45122266640277)},
    "171": {"name": "Nalli St", "coords": (17.894530576076342, 83.4477443509136)},
    "172": {"name": "Chinna Bazar Road", "coords": (17.8950587195666, 83.4449501096257)},
    "173": {"name": "Narsipatnam Road", "coords": (17.89292686675069, 83.42095344683074)},
    "174": {"name": "1st Line", "coords": (17.896328680864297, 83.44698191917901)},
    "175": {"name": "Chittivalasa Road", "coords": (17.899215691630946, 83.44505217231944)},
    "176": {"name": "Tharapuvalasa Road", "coords": (17.90822894971665, 83.43940488048837)},
    "177": {"name": "Kummaripalem", "coords": (17.90509696531714, 83.44019239127931)},
    "178": {"name": "Mamidipalem", "coords": (17.905080412994902, 83.42778970167593)},
    "179": {"name": "Mulakuddu Rural", "coords": (17.91137781491984, 83.44573805678867)},
    "180": {"name": "Jeerupeta Main Road", "coords": (17.922395931984767, 83.43476665754082)},
    "181": {"name": "Moolakaddu", "coords": (17.91605217418713, 83.44872828192948)},
    "182": {"name": "Kothapeta CJM Road", "coords": (17.92357864665624, 83.4347398816607)},
    "183": {"name": "SR Nagar", "coords": (17.926061156904982, 83.4399438727375)},
    "184": {"name": "Chittivalasa", "coords": (17.934099254769254, 83.43090621382345)},
    "185": {"name": "Kimmari St", "coords": (17.928869784214914, 83.42901505369821)},
    "186": {"name": "Kummaravidhi", "coords": (17.929446708064578, 83.42735926265738)},
    "187": {"name": "Bangalametta", "coords": (17.92999681620164, 83.4267762339267)},
    "188": {"name": "Thagarapuvalasa Main Road", "coords": (17.93266109351704, 83.4267435340209)},
    "189": {"name": "Raja Veedhi", "coords": (17.931671620416434, 83.42589268651317)},
    "190": {"name": "Joga St", "coords": (17.931112381205764, 83.42438647308364)},
    "191": {"name": "Sivalayam St", "coords": (17.93614912663304, 83.42506907349131)},
    "192": {"name": "Venkateswara St", "coords": (17.935645909475777, 83.42198207901477)},
    "193": {"name": "Thagarapuvalasa Venkateswara Swamy Temple", "coords": (17.935384446687756, 83.42193561099428)},
    "194": {"name": "Gambhiram", "coords": (17.872875046184696, 83.36878793951209)},
    "195": {"name": "Boyapalem", "coords": (17.86593131870238, 83.37424670136393)},
    "196": {"name": "Nidigattu", "coords": (17.861714967128428, 83.38766535056659)},
    "197": {"name": "Paradesipalem", "coords": (17.85759714543137, 83.36647692132058)},
    "198": {"name": "Devimetta Road", "coords": (17.845205978254267, 83.35586780354136)},
    "199": {"name": "Ozone Valley Blvd", "coords": (17.839671625523245, 83.3507341734025)},
    "200": {"name": "Madhurawada Hill Road", "coords": (17.837124537149492, 83.33394170727304)},
    "201": {"name": "Kommadi", "coords": (17.844954210572038, 83.31997767214675)},
    "202": {"name": "Kommadi 100Feet Road", "coords": (17.832040195079117, 83.334959498279)},
    "203": {"name": "Kommadi Village", "coords": (17.83015526810118, 83.3363835053488)},
    "204": {"name": "Brahmmagupta Road", "coords": (17.8248575746361, 83.3330048124499)},
    "205": {"name": "Gandhi Nagar", "coords": (17.823964098540046, 83.3456362662637)},
    "206": {"name": "Kala Nagar Road", "coords": (17.823091640821435, 83.35038410349651)},
    "207": {"name": "Chinni Gantyada", "coords": (17.821409992790905, 83.31463547498977)},
    "208": {"name": "Sector 2", "coords": (17.828922950515633, 83.35223606233636)},
    "209": {"name": "Marikavalsa", "coords": (17.826825869412257, 83.37136954226895)},
    "210": {"name": "Madhurawada", "coords": (17.824242759811934, 83.3564709407632)},
    "211": {"name": "Sivasakthi Nagar", "coords": (17.824208527347107, 83.37050830682414)},
    "212": {"name": "Kothapalem", "coords": (17.82127313337237, 83.37816447059615)},
    "213": {"name": "Ayodhyanagar", "coords": (17.819822517801903, 83.36918774730594)},
    "214": {"name": "Port Colony", "coords": (17.817667595746755, 83.36204390793213)},
    "215": {"name": "Nagarapalem Road", "coords": (17.81728702222001, 83.35834995960862)},
    "216": {"name": "Revallapalem", "coords": (17.817687931377552, 83.3473424229076)},
    "217": {"name": "Bakkanapalem Road", "coords": (17.815240236992615, 83.34341029571813)},
    "218": {"name": "Chandrampalem", "coords": (17.812716442840866, 83.35738251032842)},
    "219": {"name": "Vambay Colony", "coords": (17.81357682706591, 83.36485698742061)},
    "220": {"name": "Sampath Nagar", "coords": (17.809771163108625, 83.35295698817414)},
    "221": {"name": "Srinivasa Nagar", "coords": (17.81097705337259, 83.36085626088139)},
    "222": {"name": "Mithilapuri Colony", "coords": (17.809854216346668, 83.36683220656155)},
    "223": {"name": "Kommadi Main Road", "coords": (17.807646201734663, 83.35461119946619)},
    "224": {"name": "Durga Nagar", "coords": (17.80642537640502, 83.35679578338649)},
    "225": {"name": "Ashok Nagar", "coords": (17.80292150049234, 83.34312438767952)},
    "226": {"name": "Pilakavanipalem", "coords": (17.801449282736595, 83.35672997015394)},
    "227": {"name": "Pothinamallayya Palem", "coords": (17.799849622759478, 83.35301976524988)},
    "228": {"name": "Malatamba Road", "coords": (17.799551212437866, 83.35187741799513)},
    "229": {"name": "Tharakarama Nagar Road", "coords": (17.799959551876785, 83.35453422627552)},
    "230": {"name": "Law College Road", "coords": (17.791261881788227, 83.35341603036363)},
    "231": {"name": "Sarojini Naidu Road", "coords": (17.792454975709774, 83.36216750377872)},
    "232": {"name": "Srinivasa Nagar", "coords": (17.77911087748026, 83.3632026706977)},
    "233": {"name": "Yendada", "coords": (17.775234608170283, 83.3632898616165)},
    "234": {"name": "Pineapple Colony", "coords": (17.778915350568724, 83.27127687323706)},
    "235": {"name": "Simhachalam", "coords": (17.771371495081848, 83.23565104578613)},
    "236": {"name": "Prahaladapuram", "coords": (17.762113324887373, 83.22276413857614)},
    "237": {"name": "Gopalapatnam", "coords": (17.74852785019871, 83.21952849481089)},
    "238": {"name": "NAD Junction", "coords": (17.743817571842417, 83.231596390193)},
    "239": {"name": "Madhavadhara", "coords": (17.74870940055509, 83.25802885492473)},
    "240": {"name": "LB Nagar", "coords": (17.738728477242915, 83.27308655100138)},
    "241": {"name": "Bapuji Nagar", "coords": (17.7404191806239, 83.27838073103622)},
    "242": {"name": "Kancharapalem", "coords": (17.73550817777932, 83.27357186873668)},
    "243": {"name": "Dharma Nagar", "coords": (17.73346890704617, 83.28249007307599)},
    "244": {"name": "Boyapalem", "coords": (17.73241783813577, 83.28377530333609)},
    "245": {"name": "Santhi Nagar", "coords": (17.739260341508118, 83.28678343562495)},
    "246": {"name": "Kailasapuram", "coords": (17.74070300271217, 83.2890244069632)},
    "247": {"name": "Akkayapalem Junction", "coords": (17.74060709263041, 83.3000317823774)},
    "248": {"name": "Vizag - Srikakulam Highway", "coords": (17.738759056001435, 83.3037387704069)},
    "249": {"name": "Abid Nagar", "coords": (17.73882098009648, 83.2990941418262)},
    "250": {"name": "Akkayapalem 80 Feet Road", "coords": (17.735092021304435, 83.29254710805513)},
    "251": {"name": "Akkayapalem", "coords": (17.735428884410688, 83.29980148531826)},
    "252": {"name": "Nandagiri Nagar", "coords": (17.732686530936025, 83.29379116969619)},
    "253": {"name": "NT College Road", "coords": (17.733249215298667, 83.30063713635944)},
    "254": {"name": "Jagannadhapuram Junction", "coords": (17.73001174656923, 83.29920142377334)},
    "255": {"name": "Akkayapalem Main Road", "coords": (17.728754496223374, 83.29914642754274)},
    "256": {"name": "Railway New Colony", "coords": (17.72713053810997, 83.29879577561624)},
    "257": {"name": "Telugu Thalli Flyover", "coords": (17.72527971698318, 83.2993956372019)},
    "258": {"name": "Diamond Park Road", "coords": (17.72243489154749, 83.30320902502436)},
    "259": {"name": "Asilmetta Junction", "coords": (17.724583491237482, 83.30857267160535)},
    "260": {"name": "Jail Road", "coords": (17.722052953648433, 83.3076871292065)},
    "261": {"name": "RamNagar Road", "coords": (17.718630234415578, 83.30813005437619)},
    "262": {"name": "NTR Road", "coords": (17.723975863391605, 83.31077172469244)},
    "263": {"name": "Sampath Vinayaka Temple Road", "coords": (17.72473550914981, 83.31340632751815)},
    "264": {"name": "Siripuram Circle", "coords": (17.723817603154, 83.31711375959895)},
    "265": {"name": "Waltair Main Road", "coords": (17.722198821877395, 83.31655835656764)},
    "266": {"name": "RTC Complex Road", "coords": (17.72323156497607, 83.3060858741754)},
    "267": {"name": "Daba Garden Road", "coords": (17.720258059719768, 83.30308554326034)},
    "268": {"name": "Saraswati Junction", "coords": (17.714205263699387, 83.30039728954522)},
    "269": {"name": "Jagadamba Theatre", "coords": (17.712387426239825, 83.30215326461173)},
    "270": {"name": "Leelamahal Junction", "coords": (17.714118089521982, 83.29896968440603)},
    "271": {"name": "Suryabadh Junction", "coords": (17.713298650183624, 83.29893307914605)},
    "272": {"name": "75 Feet Road", "coords": (17.7110379081839, 83.29593144769531)},
    "273": {"name": "Seethammadara Road", "coords": (17.745584952880414, 83.30685437170811)},
    "274": {"name": "Seethammadara", "coords": (17.742235027826506, 83.3092809992481)},
    "275": {"name": "NRI Hospital Road", "coords": (17.73744307891219, 83.30819597831854)},
    "276": {"name": "Gurudwara Junction", "coords": (17.736821500967988, 83.30769890449918)},
    "277": {"name": "Sankara Matham Road", "coords": (17.73609953799343, 83.30450752644661)},
    "278": {"name": "Seethammapeta Junction", "coords": (17.732130811036576, 83.30691788065958)},
    "279": {"name": "Dwaraka Nagar", "coords": (17.729813335777596, 83.3091316004016)},
    "280": {"name": "CMR Central Mall", "coords": (17.734450537781576, 83.31818195988224)},
    "281": {"name": "Resuvanipalem Road", "coords": (17.733921487922608, 83.31877707911121)},
    "282": {"name": "Thatichetlapalem", "coords": (17.734300206130158, 83.28987122925354)},
    "283": {"name": "Thatichetlapalem Road", "coords": (17.733421404124872, 83.28802594020915)},
    "284": {"name": "Railway Stadium", "coords": (17.725956235462615, 83.29100556889415)},
    "285": {"name": "Vizag Railway Station", "coords": (17.721920037352003, 83.29033631983815)},
    "286": {"name": "Chavulamadum Junction", "coords": (17.71884512550107, 83.2922444852456)},
    "287": {"name": "Allipuram Main Road", "coords": (17.7192187474021, 83.29285443133809)},
    "288": {"name": "Port Main Road", "coords": (17.71675546254211, 83.28906335920807)},
    "289": {"name": "Bowdara Road", "coords": (17.711562244851272, 83.29418746149676)},
    "290": {"name": "Town Kotha Road", "coords": (17.703442893887466, 83.2970105770434)},
    "291": {"name": "Shri Kanaka Maha Lakshmi Temple", "coords": (17.69984506583597, 83.2969133894411)},
    "292": {"name": "Kurupam Market", "coords": (17.698727349626047, 83.29491589134004)},
    "293": {"name": "Old Post Office Junction", "coords": (17.69359818156831, 83.29210667231747)},
    "294": {"name": "Andhra University", "coords": (17.728839441348253, 83.32420724880205)},
    "295": {"name": "Maddilapalem", "coords": (17.73688314771312, 83.32186017050113)},
    "296": {"name": "Jagadamba Junction", "coords": (17.7075970423274, 83.30003479475056)},
    "297": {"name": "Daba Gardens", "coords": (17.715977669760697, 83.29741695883656)},
    "298": {"name": "Satyam Junction", "coords": (17.734375953169263, 83.31298951592622)},
    "299": {"name": "Anandapuram", "coords": (17.903526612989545, 83.36990105923728)}
},

    "tourism_places": {
        "1": {"name": "RK Beach", "coords": (17.71141942032588, 83.31823543660929), "image": "rk_beach.jpg"},
        "3": {"name": "Simhachalam Temple", "coords": (17.7664, 83.2505), "image": "simhachalam.jpg"},
        "4": {"name": "Kailasagiri", "coords": (17.748992, 83.342236), "image": "kailasagiri.jpg"},
        "7": {"name": "Fishing Harbour", "coords": (17.69594653743959, 83.3025196957754), "image": "Fishing_Harbour.jpg"},
        "8": {"name": "Tenneti Park", "coords": (17.747944, 83.349915), "image": "tenneti_park.jpg"},
        "9": {"name": "Rushikonda Beach", "coords": (17.78252, 83.385115), "image": "rushikonda_beach.jpg"},
        "10": {"name": "INS Kursura Submarine Museum", "coords": (17.7175, 83.329444), "image": "ins_kursura_submarine.jpg"},
        "11": {"name": "Indira Gandhi Zoological Park", "coords": (17.765667, 83.348576), "image": "zoo_park.jpg"},
        "12": {"name": "Visakha Museum", "coords": (17.720707, 83.333831), "image": "visakha_museum.jpg"},
        "14": {"name": "Yarada Beach", "coords": (17.6644, 83.2844), "image": "yarada_beach.jpg"},
        "15": {"name": "VUDA Park", "coords": (17.72408, 83.339281), "image": "vuda_park.jpg"},
        "16": {"name": "Kambalakonda Wildlife Sanctuary", "coords": (17.825278, 83.308611), "image": "kambalakonda.jpg"},
        "17": {"name": "Thotlakonda Buddhist Complex", "coords": (17.826389, 83.409444), "image": "thotlakonda.jpg"},
        "18": {"name": "Dolphin's Nose", "coords": (17.826389, 83.409444), "image": "dolphins_nose.jpg"},
        "19": {"name": "Matsyadarsini Aquarium", "coords": (17.712825525387323, 83.32035001459779), "image": "matsyadarsini.jpg"},
        "20": {"name": "TU 142 Aircraft Museum", "coords": (17.718002, 83.329812), "image": "tu_142_museum.jpg"},
        "21": {"name": "Erra Matti Dibbalu", "coords": (17.874948, 83.431598), "image": "erra_matti_dibbalu.jpg"},
        "22": {"name": "Biodiversity Park", "coords": (17.72903, 83.336432), "image": "biodiversity_park.jpg"},
        "23": {"name": "Mudasarlova Park", "coords": (17.765346, 83.294556), "image": "mudasarlova_park.jpg"},
        "24": {"name": "Sagar Nagar Beach", "coords": (17.763611267955948, 83.36149055475887), "image": "sagar_nagar_beach.jpg"},
        "25": {"name": "Gangavaram Beach", "coords": (17.61931720547679, 83.23271672751862), "image": "gangavaram_beach.jpg"},
        "26": {"name": "Ross Hill Church", "coords": (17.69042212498955, 83.28713355089681), "image": "ross_hill_church.jpg"},
        "27": {"name": "Kondakarla Ava", "coords": (17.600852, 82.998148), "image": "kondakarla_ava.jpg"},
        "28": {"name": "Shilparamam(jatara)", "coords": (17.805223113037975, 83.35317529554837), "image": "shilparamam.jpg"},
        "29": {"name": "Bheemili Lighthouse","coords": (17.89085867334386, 83.45571126671453), "image": "bheemili_lighthouse.jpg"},
        "30": {"name": "Bheemili Beach","coords": (17.888931060832707, 83.45251238641332), "image": "bheemili_beach.jpg"},
        "31": {"name": "Peda Narava","coords": (17.744941226769274, 83.18190312313317), "image": "peda_narava.jpg"},
        "32": {"name": "Central Park", "coords": (17.722874036937824, 83.30663990903874),"image": "Central_park.jpg"},
        "33": {"name": "Vizag View Point", "coords": (17.756603006467486, 83.35377464367298),"image": "view_point.jpg"},
        "34": {"name": "Sri Venkateswara Swamy Devalayam, TTD", "coords": (17.77901433467497, 83.37753376012896),"image": "ttd.jpg"},
        "35": {"name": "ISKCON Temple", "coords": (17.76825004313434, 83.36665424289905),"image": "iskcon.jpg"},
        "36": {"name": "Sri Kalimatha Temple", "coords": (17.71371028779316, 83.31912069657764),"image": "skml.jpg"},
        "37": {"name": "Shivaji Park", "coords": (17.737555794394183, 83.33112379871247),"image": "Shivaji_Park.jpg"},
        "38": {"name": "Bojjanna Konda", "coords": (17.711019815970836, 83.01546887417979),"image": "Bojjanna_Konda.jpg"},
        "39": {"name": "Sri Mogadaramma Mahalakshmi Padmalayam", "coords": (17.765795776354647, 83.36415557230085),"image": "Mogadaramma.jpg"},
        "40": {"name": "Revupolavaram", "coords": (17.40604818603682, 82.81867317503664),"image": "Revupolavaram.jpg"},
    },
    "accommodations": {
         "1": {
            "name": "Feel Like Home RK beach",
            "coords": (17.717219019362034, 83.32227660994195),
            "address": "Kamala priya residency, 7-5-111, near RK beach, Pandurangapuram, Visakhapatnam, Andhra Pradesh 530003",
            "image": "Feel_Like_Home_RK_beach.jpg"
        },
        "2": {
            "name": "Hotel KBG villa Homes -BEACH VIEW",
            "coords": (17.71023198443451, 83.31439328729233),
            "address": "HOTEL INOX, DOOR NO 15-5-3 AND 15-5-3/1 P867+2PR Visakhapatnam, Andhra Pradesh NOVTEL, THREATRE ROAD, nearby Country Club, Maharani Peta, Visakhapatnam, Andhra Pradesh 530001",
            "image": "Hotel_KBG_villa_HomesBEACHVIEW.jpeg"
        },
        "3": {
            "name": "SKML Beach Guest House",
            "coords": (17.70932888743718, 83.314587593695),
            "address": "P857+PRX, Venkateswara Metta Temple Road, Krishna Nagar, Maharani Peta, Visakhapatnam, Andhra Pradesh 530002",
            "image": "SKML_Beach_Guest_House.jpeg"
        },
        "4": {
            "name": "Casa Woods",
            "coords": (118.31471829014709, 82.89195701090443),
            "address": "Araku - Visakhapatnam Rd, Ravvalaguda, Araku Valley, Andhra Pradesh 531149",
            "image": "casa_woods.jpeg"
        },
        "5": {
            "name": "USHODAYA RESORTS",
            "coords": (18.339479356737964, 82.87586541512557),
            "address": "8VQG+G3X, Padmapuram, Araku - Visakhapatnam Rd, Araku Valley, Visakhapatnam, Andhra Pradesh 531149",
            "image": "USHODAYA_RESORTS.jpeg"
        },
        "6": {
            "name": "Sun N Shine Resort",
            "coords": (18.315757459555062, 82.89212723603842),
            "address": "Public School, Near, Alluri Sitarama Raju Marg, Araku Valley, Andhra Pradesh 531151",
            "image": "Sun_N_Shine_Resort.jpeg"
        },
        "7": {
            "name": "Ion Digital Rooms for Rent",
            "coords": (117.84638918056769, 83.20902549472339),
            "address": "14-93, opp. Pendurthi Rythu Bazar, Sarada Nagar, Karmika Nagar, Pendurthi, Visakhapatnam, Andhra Pradesh 531173",
            "image": "Ion_Digital_Rooms_for_Rent.jpeg"
        },
        "8": {
            "name": "Nrk royal suite",
            "coords": (17.74826565622559, 83.26288810903934),
            "address": "Sri Sai nilayam, 39-8-30/1, Muralinagar, Madhavadhara, Visakhapatnam, Andhra Pradesh 530008",
            "image": "Nrk_royal_suite.jpeg"
        },
        "9": {
            "name": "Shree Lakshmi Guest House & Function Halls",
            "coords": (17.744200122834634, 83.22976983787507),
            "address": "Shree Complex 58-1-348 Main Road, near Baaji Junction, Nad Junction, Buchirajupalem, Visakhapatnam, Andhra Pradesh 530027",
            "image": "Shree_Lakshmi_Guest_House_&_Function_Halls.jpeg"
        },
        "10": {
            "name": "Rosee Wood",
            "coords": (17.741179651866545, 83.33811726131535),
            "address": "Rosee Wood 2-36-4, Rosee Wood, beside canara bank, Sector 10, MVP Colony, Visakhapatnam, Andhra Pradesh 530017",
            "image": "Rosee_Wood.jpeg"
        },
        "11": {
            "name": "hotel kp suites vizag",
            "coords": (17.737909669501487, 83.33659376662503),
            "address": "Plot No 19, MVP Sector 12, Sector 12, MVP Colony, Visakhapatnam, Andhra Pradesh 530017",
            "image": "hotel_kp_suites_vizag.jpeg"
        },
        "12": {
            "name": "Lotus Park Hotel",
            "coords": (17.740382599167685, 83.33419050739522),
            "address": "Plot No 19, MVP Sector 7, Sector 5, MVP Colony, Visakhapatnam, Andhra Pradesh 530017",
            "image": "Lotus_Park_Hotel.jpeg"
        },
        "13": {
            "name": "Hotel Golden Arrow - Beach Road",
            "coords": (17.79496488151275, 83.38564554312723),
            "address": "9-69, Beach Rd, near Govt. High School, Pedda Rushikonda, Rushikonda, Visakhapatnam, Andhra Pradesh 530045",
            "image": "Hotel_Golden_Arrow - Beach Road.jpeg"
        },
        "14": {
            "name": "The Capital Hotel",
            "coords": (17.801767995157803, 83.38696652438404),
            "address": "IT SEZ Road, Near Adithya Ocean Heights, Pedda Rushikonda, Rushikonda, Visakhapatnam, Andhra Pradesh 530045",
            "image": "The_Capital_Hotel.jpeg"
        },
        "15": {
            "name": "Vizag Homestay Madhurawada Sweet Home",
            "coords": (17.808547723065892, 83.35478603389858),
            "address": "IT HUB, Annapurna Arcade, Flat No. 206 100 Feet, Opp. Shilparamam Jatara, Main Rd, Midhilapuri Vuda Colony, Andhra Pradesh 530041",
            "image": "Vizag_Homestay_Madhurawada_Sweet_Home.jpeg"
        },
        "16": {
            "name": "Treebo Imperia, Kommadi",
            "coords": (17.826696753539, 83.35624246856385),
            "address": "STBL Cine World Complex, Eater stop, Srinivasa Nagar, Madhurawada, Visakhapatnam, Andhra Pradesh 530048",
            "image": "Treebo_Imperia,_Kommadi.jpeg"
        },
        "17": {
            "name": "The Invitation 365 hotel | the best restaurant in madhurawada| best hotel in vizag",
            "coords": (17.798286720739036, 83.3553148685632),
            "address": "Plot no133 to140, tharaka rama layout, opposite to vizag stadium, Madhurawada, Visakhapatnam, A4ndhra Pradesh 530041",
            "image": "The_Invitation_365.jpeg"
        },
        "18": {
            "name": "Hotel Apna Taj Residency",
            "coords": (17.68814879191094, 83.21312775440168),
            "address": "GBR Function Hall, 10-9-41, Simhagiri Hospital Road, New Gajuwaka, Gajuwaka, Visakhapatnam, Andhra Pradesh 530026",
            "image": "Hotel_Apna_Taj_Residency.jpeg"
        },
        "19": {
            "name": "Yarada Jungle Beach Resorts",
            "coords": (17.65308923479395, 83.25929632199338),
            "address": "M734+4WR, Yarada Village, Yarada, Visakhapatnam, Andhra Pradesh 530005",
            "image": "Yarada_Jungle_Beach_Resorts.jpeg"
        },
        "20": {
            "name": "Hotel O Gajuwaka Near kurmanpalem formerly SBT",
            "coords": (17.68359085245479, 83.19980029817438),
            "address": "ward no 61, 26-23-15, NH-5, Old Gajuwaka, Visakhapatnam, Andhra Pradesh 530026",
            "image": "Hotel_kurmanpalemformerly_SBT.jpeg"
        },
        "21": {
            "name": "PVR Bheemili Beach Stay",
            "coords": (17.89223357179533, 83.4530088513712),
            "address": "opposite to Bus Stand, Bheemunipatnam, Visakhapatnam, Andhra Pradesh 531163",
            "image": "PVR _Bheemili _Beach _Stay.jpeg"
        },
        "22": {
            "name": "BANANA HOMES",
            "coords": (17.91706137579293, 83.45290260341834),
            "address": "PLOT NO 34,DOLPHIN ENCLAVE, Bheemili, Andhra Pradesh 531163",
            "image": "BANANA _HOMES.jpeg"
        },
        "23": {
            "name": "Casa Beach Front, Bheemili Visakhapatnam",
            "coords": (17.888311859172568, 83.437109757051),
            "address": "Bheemunipatnam, Visakhapatnam, Andhra Pradesh 531163",
            "image": "Casa_Beach_Front,_Bheemili_Visakhapatnam.jpeg"
        },
        "24": {
            "name": "The Park, Visakhapatnam",
            "coords": (17.721595600162196, 83.3362004108896),
            "address": "Beach Rd, Lawsons Bay Colony, Pedda Waltair, Visakhapatnam, Andhra Pradesh 530023",
            "image": "The_Park_Visakhapatnam.jpg"
        },
        "25": {
            "name": "Novotel Visakhapatnam Varun Beach",
            "coords": (17.711094789685387, 83.31612620903843),
            "address": "Beach Rd, Krishna Nagar, Maharani Peta, Visakhapatnam, Andhra Pradesh 530002",
            "image": "Novotel_Visakhapatnam_Varun_Beach.jpeg"
        },
        "26": {
            "name": "Rest N' Refresh Homes",
            "coords": (17.804346726761054, 83.35765199023473),
            "address": "1-20, beside ration depot, near Car Shed Road, pillakomativanipalem, Pothinamallayya Palem, Visakhapatnam, Andhra Pradesh 530041",
            "image": "Rest_N'_Refresh_Homes.jpeg"
        },
        "27": {
            "name": "Super Hotel O Adarsh Nagar Near Kavya Hospital",
            "coords": (17.76515837210261, 83.32729535375199),
            "address": "hotel Hotel, 3-6, 4-315/2, Old Dairy Farm Rd, beside Kavya Hospital, Adarsh Nagar, Visakhapatnam, Andhra Pradesh 530040",
            "image": "superNagar_Near_Kavya_Hospital.jpeg"
        },
        "28": {
            "name": "Ns Beach Resort",
            "coords": (17.831201152299325, 83.41404296671308),
            "address": "mangamaripeta , thotlakonda, Plot no - 275, opposite Ananya hatcheries, Mangamari Peta, Visakhapatnam, Andhra Pradesh 530045",
            "image": "Ns_Beach_Resort.jpeg"
        },
        "29": {
            "name": "The Q Hotel",
            "coords": (17.71328200638072, 83.31925143999324),
            "address": "7-5-1/58, near Mastyadarshani, Pandurangapuram, Visakhapatnam, Andhra Pradesh 530003",
            "image": "The_Q_Hotel.jpeg"
        },
        "29": {
            "name": "Hotel Ocean View",
            "coords": (17.76923261735778, 83.3596171082578),
            "address": "near Pepsi Godown, Musalayya Palem, Visakhapatnam, Andhra Pradesh 530045",
            "image": "Hotel_Ocean_View.jpeg"
        }, "30": {
            "name": "Bay Leaf Resort",
            "coords": (17.7630771839891, 83.35579087673348),
            "address": "A Beach Road Beside Zoo Back Gate Ocean Drive, Sagar Nagar, Visakhapatnam, Andhra Pradesh 530045",
            "image": "Bay_Leaf_Resort.jpeg"
        },"31": {
            "name": "Treebo Excella",
            "coords": (17.71328200638072, 83.31925143999324),
            "address": "386 Revenue Employs Co-operative Society Colony, Vishalakshi Nagar, Visakhapatnam, Andhra Pradesh 530043",
            "image": "Treebo_Excella.jpeg"
        },"32": {
            "name": "The Pioneer",
            "coords": (17.737019030814178, 83.31443062900988),
            "address": "9-29, 18/1, VIP Rd, Balaji Nagar, Siripuram, Visakhapatnam, Andhra Pradesh 530003",
            "image": "The_Pioneer.jpeg"
        },"33": {
            "name": "Classic Luxury Service Apartments",
            "coords": (17.731786908854268, 83.32541695691758),
            "address": "20, 7, Kirlampudi Main Rd, Kirlampudi Layout, Chinna Waltair, Pedda Waltair, Visakhapatnam, Andhra Pradesh 530003",
            "image": "Classic_Luxury_Service_Apartments.jpeg"
        },"34": {
            "name": "Hotel Lorven",
            "coords": (17.76729982552857, 83.31002611093214),
            "address": "Chinna Gadhili, Hanumanthavaka, Visakhapatnam, Andhra Pradesh 530040",
            "image": "Hotel_Lorven.jpeg"
        },"34": {
            "name": "vasudev comforts & lodge",
            "coords": (17.764520749974995, 83.31448930664465),
            "address": "18-461,chinagadili, Health city, near sai baba temple, beside chinagadili muncipal school, Chinna Gadhili, Hanumanthavaka, Visakhapatnam, Andhra Pradesh 530040",
            "image": "vasudev_comforts_&_lodge.jpeg"
        },"36": {
            "name": "Fantasea",
            "coords": (17.765515784591354, 83.36074040611412),
            "address": "Ntr Marg, Sagar Nagar, Visakhapatnam, Andhra Pradesh 530045",
            "image": "Fantasea.jpeg"
        },"37": {
            "name": "Super Townhouse Gajuwaka Centre",
            "coords": (17.6843607754313, 83.20609695268021),
            "address": "10-2-5/2, MAINROAD, Old Gajuwaka, Chaitanya Nagar, Gajuwaka, Visakhapatnam, Andhra Pradesh 530026",
            "image": "Super_Townhouse_Gajuwaka_Centre.jpeg"
        },"38": {
            "name": "Swagan Homestays",
            "coords": (17.715731032576315, 83.30090041298185),
            "address": "28-19-14, Prakasaraopeta, Suryabagh, Jagadamba Junction, Visakhapatnam, Andhra Pradesh 530002",
            "image": "Swagan_Homestays.jpeg"
        },"39": {
            "name": "Hotel Siri Grand",
            "coords": (17.713932294285396, 83.29926962993306),
            "address": "28-11-8, Leelamahal Rd, Suryabagh, Allipuram, Visakhapatnam, Andhra Pradesh 530020",
            "image": "Hotel_Siri_Grand.jpeg"
        },"40": {
            "name": "Advith suites",
            "coords": (17.717366233551154, 83.29944129130661),
            "address": "Daba Gardens, Allipuram, Visakhapatnam, Andhra Pradesh 530020",
            "image": "Advith_suites.jpeg"
        },"41": {
            "name": "Nature's Nest Araku",
            "coords": (18.327014331157947, 82.89638057594713),
            "address": "Main road Araku ,Ravvalaguda, Visakhapatnam, Andhra Pradesh 531149",
            "image": "katikiaccm.jpg"
        }
    },
    "restaurants" : {
        "1": {
            "name": "Aha Yemi Ruchulu The Kitchen",
            "coords": (17.739936035734722, 83.34124538068401),
            "address": "plot no 126, opposite shankar car garage, Sector 10, MVP Colony, Visakhapatnam, Andhra Pradesh 530017",
            "image": "aha.jpg"
        },
        "2": {
            "name": "Sivakoti's Food Magic Restaurant",
            "coords": (17.754137778648793, 83.34090419982682),
            "address": "10-21-14, Police Quarters Rd, opp. HP Petrol Bunk, Vishalakshi Nagar, Visakhapatnam, Andhra Pradesh 530043",
            "image": "sivakoti.jpg"
        },
        "3": {
            "name": "Hotel Kamat",
            "coords": (17.734432975950646, 83.34115365749854),
            "address": "Beach Rd, Lawsons Bay Colony, Pedda Waltair, Visakhapatnam, Andhra Pradesh 530017",
            "image": "kamat.jpg"
        },
        "4": {
            "name": "WelcomCafe Oceanic Restaurant",
            "coords": (17.71228682082343, 83.31537633062489),
            "address": "WelcomHotel Grand Bay, Beach Rd, Krishna Nagar, Maharani Peta, Visakhapatnam, Andhra Pradesh 530002",
            "image": "WelcomCafe.jpg"
        },
        "5": {
            "name": "Food World",
            "coords": (17.712341582140652, 83.3182204427351),
            "address": "P869+R83 RK Beech, Pandurangapuram, Visakhapatnam, Andhra Pradesh 530002",
            "image": "Food_World.jpg"
        },
        "6": {
            "name": "Varun's Eat Restaurant (Varun Beach Inox)",
            "coords": (17.711028946524817, 83.31582178684513),
            "address": "I st. Floor, Varun Inox, opp. I.B.P.Century Club, Krishna Nagar, Maharani Peta, Visakhapatnam, Andhra Pradesh 530002",
            "image": "Varun_Beach_Inox.jpg"
        },
        "7": {
            "name": "Naidu Hotel",
            "coords": (18.275770223161164, 83.03709695275928),
            "address": "Rampalle, Andhra Pradesh 531149",
            "image": "Naidu_Hotel.jpg"
        },
        "8": {
            "name": "Sri Sairam Hotel",
            "coords": (18.275939836303795, 83.03717350560682),
            "address": "72GP+8VC, near Borra caves, Andhra Pradesh 531149",
            "image": "Sri_Sairam_Hotel.jpg"
        },
        "9": {
            "name": "Sri Anjaneya Bamboo Chicken",
            "coords": (18.279186684491467, 83.03770937553968),
            "address": "72HQ+H4C, Borra, Post, Ananthagiri, Andhra Pradesh 531149",
            "image": "Sri_Anjaneya_Bamboo_Chicken.jpg"
        },
        "10": {
            "name": "Sri Sai Balaji food parlour &best quality catering",
            "coords": (17.772453507822174, 83.24208547422762),
            "address": "SHOP NO 2 & 3 SRIDEVI NEW SHOPPING COMPLEX, Old Adavivaram, Sri Sai Nagar, Simhachalam, Visakhapatnam, Andhra Pradesh 530028",
            "image": "Sri_Sai_Balaji.jpg"
        },
        "11": {
            "name": "Sitara grand restaurant",
            "coords": (17.772208304079264, 83.23839475469612),
            "address": "Q6CQ+F8H, Old Adavivaram, Sri Sai Nagar, Simhachalam, Visakhapatnam, Andhra Pradesh 530028",
            "image": "Sitara_grand_restaurant.jpg"
        },
        "12": {
            "name": "Vatan",
            "coords": (17.773883856285128, 83.23127080769345),
            "address": "24-168, Srinivas Nagar, Prahaladapuram, Simhachalam, Visakhapatnam, Andhra Pradesh 530028",
            "image": "Vatan.jpg"
        },
        "13": {
            "name": "Royal Darbar Restaurant",
            "coords": (18.322252692085435, 82.88089370885135),
            "address": "MAIN ROAD, beside MPTO OFFICE, Araku Valley, Andhra Pradesh 531151",
            "image": "Royal_Darbar_Restaurant.jpg"
        },
        "14": {
            "name": "Hotel Star Annapurna multi cuisine",
            "coords": (18.322154460732808, 82.88013306088236),
            "address": "Araku Valley, Andhra Pradesh 531149",
            "image": "Hotel_Star_Annapurna_multi_cuisine.jpg"
        },
        "15": {
            "name": "MTR RESTAURANT ARAKU",
            "coords": (18.33345918960708, 82.87536952957149),
            "address": "Junction, beside MTR Resorts, Padmapuram, Araku Valley, Andhra Pradesh 531149",
            "image": "MTR_RESTAURANT.jpg"
        },
        "16": {
            "name": "Naidu Gari Kunda Biryani - Rushikonda",
            "coords": (17.782896358092803, 83.38339140519838),
            "address": "Junction, beside MTR Resorts, Padmapuram, Araku Valley, Andhra Pradesh 531149",
            "image": "Naidu_Gari.jpg"
        },
        "17": {
            "name": "Shore Front Resort",
            "coords": (17.78321276447879, 83.38443463774419),
            "address": "Rushikonda, Visakhapatnam, Andhra Pradesh 530045",
            "image": "Shore_Front_Resort.jpg"
        },
        "18": {
            "name": "Rushi Valley Restaurant",
            "coords": (17.79240710823705, 83.38735288109469),
            "address": "D.no:7-72/1, near Sri Sri Sri polamamba ammavarla aalayam, Pedda Rushikonda, Rushikonda, Visakhapatnam, Andhra Pradesh 530040",
            "image": "Rushi_Valley_Restaurant.jpg"
        },
        "19": {
            "name": "Reboot Dine In",
            "coords": (17.7619091237445, 83.35804272460656),
            "address": "3-11 Beach Road Near Sagar Nagar Geetam College Post, Gudla Vani Palem, Visakhapatnam, Andhra Pradesh 530045",
            "image": "Reboot_Dine_In.jpg"
        },
        "20": {
            "name": "Sandhya Bangali Hotel",
            "coords": (17.76339658204958, 83.35907629103001),
            "address": "Beach Rd, Sagar Nagar, Visakhapatnam, Andhra Pradesh 530045",
            "image": "Sandhya_Bangali_Hotel.jpg"
        },
        "21": {
            "name": "Masalah Mafia @ Palm Beach Hotel",
            "coords": (17.720787083028124, 83.33589623394403),
            "address": "505, Beach Rd, Kirlampudi Layout, Chinna Waltair, Pandurangapuram, Visakhapatnam, Andhra Pradesh 530017",
            "image": "Masalah_Mafia.jpg"
        },
        "22": {
            "name": "Taj's Korean street food truck",
            "coords": (17.72431263575267, 83.33812319563611),
            "address": "Beach Rd, East Point Colony, Pedda Waltair, Visakhapatnam, Andhra Pradesh 530017",
            "image": "Taj's_Korean.jpg"
        },
        "23": {
            "name": "Vasenapoli",
            "coords": (17.71873995073398, 83.3314493825876),
            "address": "RK Beach Rd, near VICTORY AT SEA, beside Hotel Ambica Sea Green, Kirlampudi Layout, Chinna Waltair, Pedda Waltair, Visakhapatnam, Andhra Pradesh 530017",
            "image": "Vasenapoli.jpg"
        },
        "24": {
            "name": "Yarada Restaurant",
            "coords": (17.659934752353543, 83.275427484739),
            "address": "Yarada Rd, Yarada, Visakhapatnam, Andhra Pradesh 530014",
            "image": "Yarada_Restaurant.jpg"
        },
        "25": {
            "name": "Amrutha Restaurant and bar",
            "coords": (17.654774648548273, 83.26344497656028),
            "address": "M727+X6M, Visakhapatnam, Andhra Pradesh 530005",
            "image": "Amrutha_Restaurant.jpg"
        },
        "26": {
            "name": "SOMAA, Yendada",
            "coords": (17.779183174971095, 83.35456258097365),
            "address": "Plot no 16, beside HPCL petrol bunk, vivekanada nagar, Yendada, Visakhapatnam, Andhra Pradesh 530045",
            "image": "SOMAA.jpg"
        },
        "27": {
            "name": "NAVASA - Fine Dining | Best Dum biryani in vizag",
            "coords": (17.778908264035827, 83.35394909588058),
            "address": "Plot No. 12, Vivekananda nagar, behind HPCL petrol Bunk, Yendada, Visakhapatnam, Andhra Pradesh 530045",
            "image": "NAVASA.jpg"
        },
        "28": {
            "name": "Kambala Konda Cafeteria",
            "coords": (17.772035353151498, 83.34077720993541),
            "address": "Kambalakonda Eco Tourism Park, Visakhapatnam, Andhra Pradesh 531173",
            "image": "Kambala_Konda_Cafeteria.jpg"
        },
        "29": {
            "name": "Haveli Restaurant",
            "coords": (17.810989447338137, 83.4060819094013),
            "address": "RC64+6C6, Thimmapuram, Visakhapatnam, Andhra Pradesh 530048",
            "image": "Haveli.jpg"
        },
        "30": {
            "name": "Sea View Restaurant ( Chowdary gari dhaba)",
            "coords": (17.809546417338936, 83.40536016223298),
            "address": "Beside Sri Sai Ganesh Hatchery, Bheemili Beach Road, Thimmapuram, Visakhapatnam, Andhra Pradesh 530048",
            "image": "Sea_View_Restaurant.jpg"
        },
        "31": {
            "name": "Sanctum Beach Resorts",
            "coords": (17.822441311702494, 83.41504719271083),
            "address": "Thotlakonda Beach, Beach Rd, Mangamari Peta, Kapuluppada, Visakhapatnam, Andhra Pradesh 530048",
            "image": "Sanctum_Beach_Resorts.jpg"
        },
        "32": {
            "name": "The Vizag Drive In ( Chinese, North Indian, Grill Tandoori, Pulaos, Shakes, Italian, Mexican, Pantry.)",
            "coords": (17.732006452599506, 83.34090611530061),
            "address": "T4-72-14, Beach Rd, next to BPCL Petrol Bunk, Lawsons Bay Colony, MVP Colony, Visakhapatnam, Andhra Pradesh 530017",
            "image": "The_Vizag_Drive_In.jpg"
        },
        "33": {
            "name": "Dine Destiny Fine Dining Restaurant",
            "coords": (17.72801178707383, 83.33929459408755),
            "address": "Hotel Ocean Vista Bay,Cricket Academy, Beach Rd, East Point Colony, Visakhapatnam, Andhra Pradesh 530017",
            "image": "Dine_Destiny.jpg"
        },
        "34": {
            "name": "ABDUL'S KITCHEN (A Multi-Cuisine Family Restaurant)",
            "coords": (17.763522824399807, 83.313754112186),
            "address": "Chinagadili, near Sai Baba Temple, Chinna Gadhili, Hanumanthavaka, Visakhapatnam, Andhra Pradesh 530040",
            "image": "ABDUL.jpg"
        },
        "35": {
            "name": "Studio Grill",
            "coords": (17.76658451639587, 83.3631535100883),
            "address": "Musalayyapalem, Sagar Nagar, Visakhapatnam, Andhra Pradesh 530045",
            "image": "Studio_Grill.jpg"
        },
        "36": {
            "name": "Radisson Blu Resort Visakhapatnam",
            "coords": (17.77156905650209, 83.37290147515961),
            "address": "Survey No: 106, Rushikonda, Beach Road, Yendada, Visakhapatnam, Andhra Pradesh 530045",
            "image": "Radisson.jpg"
        },
        "37": {
            "name": "Beach road Dosa point",
            "coords": (17.766924683049094, 83.365224752445),
            "address": "Q988+P4X, Yendada, Visakhapatnam, Andhra Pradesh 530045",
            "image": "Beach _road_Dosa_point.jpg"
        },
        "38": {
            "name": "Vizagapatam Family Restaurant",
            "coords": (17.698101704683978, 83.29875633837851),
            "address": "M7XX+2H3, Chengal Rao Peta, Port Area, Visakhapatnam, Andhra Pradesh 530001",
            "image": "Vizagapatam_Family_Restaurant.jpg"
        },
        "39": {
            "name": "Zeeshan Restaurant - Apna Hyderabadi Food",
            "coords": (17.80447480330742, 83.35308138855358),
            "address": "PNB COMPLEX, Service Road, Car Shed, Junction, NH16, Madhurawada, Visakhapatnam, Andhra Pradesh 530041",
            "image": "Zeeshan_Restaurant.jpg"
        },
        "40": {
            "name": "KVR RESTAURANT",
            "coords": (17.80419310279721, 83.35301311066033),
            "address": "Main Road, Car Shed junction, NH16, Madhurawada, Visakhapatnam, Andhra Pradesh 530041",
            "image": "KVR_RESTAURANT.jpg"
        },
        "41": {
            "name": "Nidhi Family Restaurant",
            "coords": (17.807528810661733, 83.3550136307488),
            "address": "R943+WXM, Jatara, Shilparamam, Madhurawada, Visakhapatnam, Andhra Pradesh 530041",
            "image": "Nidhi_Family_Restaurant.jpg"
        },
        "42": {
            "name": "SRI VENNELA FAMILY RESTAURANT",
            "coords": (17.892693292786678, 83.45297647283896),
            "address": "near bus stand, beside andhrabank, Peddabazar, Bheemunipatnam, Visakhapatnam, Andhra Pradesh 531163",
            "image": "SRI_VENNELA_FAMILY_RESTAURANT.jpg"
        },
        "43": {
            "name": "PFC Pearls Food Court",
            "coords": (17.891855574148973, 83.45513150826915),
            "address": "Bheemili Light House, Beach Rd, Bheemunipatnam, Visakhapatnam, Andhra Pradesh 531163",
            "image": "PFC_Pearls_Food_Court.jpg"
        },
        "44": {
            "name": "Zamindari Restaurant",
            "coords": (17.715691680922536, 83.31419239071897),
            "address": "Near, Nowroji Rd, Port Officers Quarters, Maharani Peta, Visakhapatnam, Andhra Pradesh 530002",
            "image": "Zamindar_res.jpg"
    },"45": {
            "name": "Royal Darbar Restaurant",
            "coords": (18.330452136997405, 82.88248248797359),
            "address": "MAIN ROAD, beside MPTO OFFICE, Araku Valley, Andhra Pradesh 531151",
            "image": "Royal_res.jpeg"
    }
    },
    "hospitals" : {
        "1": {
            "name": "Queens NRI Hospital",
            "coords": (17.7404353, 83.3080536),
            "address": "Queens NRI Hospital, NRI Hospital Road, Seethammadara, Visakhapatnam - 530013, Andhra Pradesh, India",
            "image": "nri.jpeg"
        },
        "2": {
            "name": "Vijetha Hospital",
            "coords": (17.7106199, 83.3030283),
            "address": "Vijetha Hospital, Chitralaya Road, Jagadamba Junction, Visakhapatnam - 530002, Andhra Pradesh, India",
            "image": "vijetha_hospital.jpeg"
        },
        "3": {
            "name": "A N Beach Hospital",
            "coords": (17.7133, 83.3203),
            "address": "Door No. 15-9, 13/24, Dr. NTR Beach Road, Pandurangapuram, Krishna Nagar, Maharani Peta, Visakhapatnam, Andhra Pradesh 530002, India",
            "image": "an.jpeg"
        },
        "4": {
            "name": "Giggles by Omni RK - Women and Children's Hospital",
            "coords": (17.7235, 83.3169),
            "address": "Beside Omni Hospitals, Waltair Main Road, Opposite Lions Club Of Visakhapatnam, Ram Nagar, Visakhapatnam, Andhra Pradesh 530002, India",
            "image": "omni_rk.jpeg"
        },
        "5": {
            "name": "Medicover Hospitals",
            "coords": (17.7165, 83.3182),
            "address": "Beside Omni Hospitals, Waltair Main Road, Opposite Lions Club Of Visakhapatnam, Ram Nagar, Visakhapatnam, Andhra Pradesh 530002, India",
            "image": "Medicover_hospital.jpeg"
        },
        "6": {
            "name": "Primary Health Centre (PHC)",
            "coords": (18.2833, 83.0500),
            "address": "Araku Visakhapatnam Road, Sunkarametta, Visakhapatnam - 531149, Andhra Pradesh, India",
            "image": "Primary_Health.jpeg"
        },
        "7": {
            "name": "Community Health Center (CHC)",
            "coords": (18.3273, 82.8775),
            "address": "8VFP+G6W, Araku Valley, Andhra Pradesh 531149, India",
            "image": "Community_Health.jpeg"
        },
        "8": {
            "name": "Area Hospital",
            "coords": (18.3330, 82.9000),
            "address": "Mppschool Beside Arakuvalley, Visakhapatnam District, Dumbriguda, Araku, Andhra Pradesh, 531151, India",
            "image": "Area_hospital.jpeg"
        },
        "9": {
            "name": "Simhachalam Government Hospital",
            "coords": (17.7669, 83.2484),
            "address": "7-32, Simhachalam Rd, Old Adavivaram, Sri Sai Nagar, Simhachalam, Visakhapatnam, Andhra Pradesh 530028, India",
            "image": "Simhachalam_Government.jpeg"
        },
        "10": {
            "name": "Simhadri Women & Child Hospital",
            "coords": (17.7669, 83.2484),
            "address": "Q68G+67C, Simhachalam Rd, Prahaladapuram, Simhachalam, Visakhapatnam, Andhra Pradesh 530027, India",
            "image": "Simhadri_Women.jpeg"
        },
        "11": {
            "name": "Suryanarayana Hospital",
            "coords": (17.7509, 83.2186),
            "address": "Pendurthi-NAD BRTS Expressway, Gopalapatnam, Simhachalam, Visakhapatnam, Andhra Pradesh 530027, India",
            "image": "Suryanarayana_Hospital.jpeg"
        },
        "12": {
            "name": "Pradhama Multispeciality Hospital",
            "coords": (17.7426, 83.3195),
            "address": "Near Venkojipalem Signal Junction, Visakhapatnam, Andhra Pradesh, India",
            "image": "Pradhama_Multispeciality.jpeg"
        },
        "13": {
            "name": "Sri Surya Hospital",
            "coords": (17.7214, 83.3230),
            "address": "15-14 9/1, Krishna Nagar Road, Srirangapuram, Krishna Nagar, Maharani Peta, Visakhapatnam, Andhra Pradesh 530002, India",
            "image": "Sri_Surya_Hospital.jpeg"
        },
        "14": {
            "name": "M.V.P. Hospital", 
            "coords": (17.74215, 83.33518),
            "address": "Double Road, M.V.P. Hospital, Visakhapatnam, Andhra Pradesh 530017, India",
            "image": "MVP_Hospital.jpeg"
        },
        "15": {
            "name": "Sai Krishna Medical and General Stores",
            "coords": (17.7386, 83.3057),
            "address": "Akkayyapalem, Visakhapatnam, Andhra Pradesh 530016, India",
            "image": "Sai_Krishna_Medical.jpeg"
        },
        "16": {
            "name": "GITAM Hospital Emergency Ward",
            "coords": (17.7825, 83.3773),
            "address": "GITAM University Campus, Rushikonda, Visakhapatnam, Andhra Pradesh 530045, India",
            "image": "GITAM_Hospital.jpeg"
        },
        "17": {
            "name": "Krishna Hospital",
            "coords": (17.7171, 83.3162),
            "address": "14-37-29/1, Z. P. Junction, Krishna Nagar, Maharanipeta, Visakhapatnam - 530002, Andhra Pradesh, India",
            "image": "Krishna_Hospital.jpeg"
        },
        "18": {
            "name": "Rani Chandramani Devi Government Hospital",
            "coords": (17.7386, 83.3057),
            "address": "Shop No. 17, Chinmaya Marg, Pedda Waltair Junction, Lawsons Bay Colony, Pedda Waltair, Visakhapatnam, Andhra Pradesh 530017, India",
            "image": "Rani_Chandramani.jpeg"
        },
        "19": {
            "name": "King George Hospitals",
            "coords": (17.7087, 83.3060),
            "address": "Opposite District Collectorate Office, Maharanipeta, Visakhapatnam, Andhra Pradesh 530002, India",
            "image": "Kgh.jpeg"
        },
        "20": {
            "name": "Kavya Hospital",
            "coords": (17.7590, 83.3425),
            "address": "4-337/1, Sundar Nagar, Arilova, Landmark: Old Dairy Farm, Visakhapatnam, Andhra Pradesh 530040, India",
            "image": "Kavya_Hospital.jpeg"
        },
        "21": {
            "name": "The General Clinic",  
            "coords": (17.7100, 83.3160),
            "address": "Unit No. 16-1-6-1, P L Plaza, Beach Approach Road, Apsara Road, Collector Office Junction, Maharani Peta, Visakhapatnam, Andhra Pradesh 530002, India",
            "image": "The_general.jpeg"
        },
        "22": {
            "name": "Starkids Children's Hospital",
            "coords": (17.7390, 83.3165),
            "address": "P8Q5+MM6, TPT Colony, Balayya Sastri Layout, Seethammadara, Visakhapatnam, Andhra Pradesh 530013, India",
            "image": "Starkids_Children's.jpeg"
        },
        "23": {
            "name": "Sri Sai Hospital",
            "coords": (17.7260, 83.3160),
            "address": "9-36-19/3, Pithapuram Colony, Dwaraka Nagar, Beside Sakunthala Nilayam, Visakhapatnam, Andhra Pradesh 530003, India",
            "image": "Sri_Sai_Hospital.jpeg"
        },
        "24": {
            "name": "CARE Hospital",
            "coords": (17.7176, 83.3161),
            "address": "10-50-11/5, AS Raja Complex, Waltair Main Road, Ram Nagar, Visakhapatnam, Andhra Pradesh 530002, India",
            "image": "CARE_Hospital.jpeg"
        },
        "25": {
            "name": "Primary Health Centre, Tajangi",
            "coords": (18.1833, 82.7000),
            "address": "Tajangi, Alluri Sitharama Raju district, Andhra Pradesh, India",
            "image": "Primary_Health_Centre,_Tajangi.jpeg"
        },
        "26": {
            "name": "Primary Health Centre, Kothapalli",
            "coords": (18.1500, 82.7500),
            "address": "Kothapalli, Alluri Sitharama Raju district, Andhra Pradesh, India",
            "image": "Primary_Health_Centre_Kothapalli.jpeg"
        },
        "27": {
            "name": "1 Town PHC EVAIDYA",
            "coords": (17.7040, 83.2970),
            "address": "Town Kotha Road, Port Area, Visakhapatnam, Andhra Pradesh, India",
            "image": "Town_PHC.jpeg"
        },
        "28": {
            "name": "Government Victoria Hospital",
            "coords": (17.6988, 83.2982),
            "address": "Chengal Rao Peta, Visakhapatnam, Andhra Pradesh, India",
            "image": "Government_Victoria_Hospital.jpeg"
        },
        "29": {
            "name": "St. Ann's Jubilee Memorial Hospital",
            "coords": (17.7355, 83.2715),
            "address": "62-2-106, Opposite Saraswathi School, Malkapuram, Visakhapatnam, Andhra Pradesh 530011, India",
            "image": "St_Ann_Jubilee_Memorial_Hospital.jpeg"
        },
        "30": {
            "name": "A.N. Beach Hospital",
            "coords": (17.7192, 83.3333),
            "address": "Door No. 15-9-13/24, Dr. NTR Beach Road, Pandurangapuram, Krishna Nagar, Maharani Peta, Visakhapatnam, Andhra Pradesh 530002, India",
            "image": "A_N_Beach_Hospital.jpeg"
        },
        "31": {
            "name": "Seven Hills Hospital",
            "coords": (17.7174, 83.3094),
            "address": "Rockdale Layout, Waltair Main Road, Ram Nagar, Visakhapatnam, Andhra Pradesh 530002, India",
            "image": "Seven_Hills_Hospital.jpeg"
        },
        "32": {
            "name": "Apollo Hospitals",
            "coords": (17.7171, 83.3091),
            "address": "10-50-80, Waltair Main Road, Opposite Daspalla, Ram Nagar, Visakhapatnam, Andhra Pradesh 530002, India",
            "image": "Apollo_Hospital.jpeg"
        },
        "33": {
            "name": "Apollo Hospitals, Visakhapatnam",
            "coords": (17.7611, 83.3210),
            "address": "Health City, Arilova, Visakhapatnam, Andhra Pradesh 530040, India",
            "image": "Apollo_Arilova.jpeg"
        },
        "34": {
            "name": "Pinnacle Hospitals",
            "coords": (17.7615, 83.3208),
            "address": "Chinna Gadhili, Hanumanthavaka, Visakhapatnam, Andhra Pradesh 530040, India",
            "image": "Pinnacle_Hospitals.jpeg"
        },
        "35": {
            "name": "Cotr Jyothi Hospital",
            "coords": (18.5333, 83.2167),
            "address": "Ramnagar Colony, Salur, Andhra Pradesh 535591, India",
            "image": "Cotr_Jyothi_Hospital.jpeg"
        },
        "36": {
            "name": "Visakha Institute of Medical Sciences (VIMS)",
            "coords": (17.7611, 83.3210),
            "address": "Hanumanthavaka, Visakhapatnam, Andhra Pradesh 530040, India",
            "image": "vims.jpeg"
        },
        "37": {
            "name": "Akarsh Hospitals",
            "coords": (17.6515, 83.1910),
            "address": "63-3-39/1, Old HP Petrol Bunk Backside, Opposite HPCL, Malkapuram, Visakhapatnam, Andhra Pradesh 530011, India",
            "image": "Akarsh_Hospitals.jpeg"
        },
        "38": {
            "name": "Kalyani Hospital",
            "coords": (17.6870, 83.2456),
            "address": "Malkapuram, Gandhigram Post, Nausena Baugh, Kalyani, Visakhapatnam, Andhra Pradesh 530005, India",
            "image": "Kalyani_Hospital.jpeg"
        },
        "39": {
            "name": "Andhra Medical College",
            "coords": (17.7042, 83.2998),
            "address": "P843+HWX, Medical College Road, King George Hospital, Opp. Collector Office, Maharan腊ipeta, Visakhapatnam, Andhra Pradesh 530002, India",
            "image": "Andhra_Medical_College.jpeg"
        },
        "40": {
            "name": "Seven Hills Hospital",
            "coords": (17.7172, 83.3095),
            "address": "Rockdale Layout, Waltair Main Road, Ramnagar, Visakhapatnam, Andhra Pradesh 530002, India",
            "image": "Seven_Hills_Hospital.jpeg"
        },
        "41": {
            "name": "Sraddha Multispeciality Hospital",
            "coords": (17.7084, 83.3087),
            "address": "Coastal Battery Rd, Srirangapuram, Krishna Nagar, Maharani Peta, Visakhapatnam, Andhra Pradesh 530002, India",
            "image": "Sraddha_Multispeciality_Hospital.jpeg"
        },
        "42": {
            "name": "Dr. Ramakrishna Hospital",
            "coords": (17.93093, 83.42606),
            "address": "Visakhapatnam, Andhra Pradesh 531163, India",
            "image": "Dr_Ramakrishna_Hospital.jpeg"
        },
        "43": {
            "name": "Arogya Hospital",
            "coords": (17.9315, 83.4272),
            "address": "Visakhapatnam, Andhra Pradesh 531163, India",
            "image": "Arogya_Hospital.jpeg"
        },
        "44": {
            "name": "Surya Nursing Home",
            "coords": (17.9323, 83.4248),
            "address": "Visakhapatnam, Andhra Pradesh 531163, India",
            "image": "Surya_Nursing_Home.jpeg"
        },
        "45": {
            "name": "GIMS Hospital",
            "coords": (17.7610, 83.3220),
            "address": "Health City, Chinagadili, Visakhapatnam, Andhra Pradesh, India",
            "image": "GIMS_Hospital.jpeg"
        },
        "46": {
            "name": "Visakha Steel General Hospital (VSGH)",
            "coords": (17.6631, 83.1440),
            "address": "Steel Plant Colony, Visakhapatnam, Andhra Pradesh 530032, India",
            "image": "Visakha_Steel.jpeg"
        },
        "47": {
            "name": "Nehru Hospital",
            "coords": (17.704722, 83.197222),
            "address": "Reddy Street, Malkapuram, Visakhapatnam, Andhra Pradesh 530011, India",
            "image": "Nehru_Hospital.jpg"
        },
        "48": {
            "name": "Devendra First Aid Clinic Center",
            "coords": (17.6210, 82.9270),
            "address": "Ompolu, Kondakarla, Visakhapatnam, Andhra Pradesh 531001, India",
            "image": "Devendra_First.jpeg"
        },
        "49": {
            "name": "Anil Neerukonda Hospital (ANH)",
            "coords": (17.8510, 83.3950),
            "address": "Opposite Three Polamamba Temple, Sangivalasa, Tagarapuvalasa, Visakhapatnam, Andhra Pradesh 531162, India",
            "image": "anh.jpeg"
        },
        "50": {
            "name": "Sri Venkateswara Hospital",
            "coords": (17.7630, 83.2485),
            "address": "17-118, Simhachalam to Hanumanthavaka Junction BRTS Road, Simhagiri Colony, Pedda Gadhili, Hanumanthavaka, Visakhapatnam, Andhra Pradesh 530040, India",
            "image": "Sri_Venkateswara.jpeg"
        },
        "51": {
            "name": "Apollo Health City Hospital, Arilova",
            "coords": (17.7510, 83.3360),
            "address": "Plot No:1, Arilova, Chinagadili, Visakhapatnam, Andhra Pradesh 530040, India",
            "image": "Apollo_Arilova.jpeg"
        },
        "52": {
            "name": "Tagarapuvalasa Clinic",
            "coords": (17.8900, 83.4400),
            "address": "Door No 10-37/40, Old Local Office, Anandavanam, Bheemunipatnam, Visakhapatnam - 531163, India",
            "image": "Tagarapuvalasa_Clinic.jpeg"
        },
        "53": {
            "name": "C.O.T.R. Jyothi Hospital",
            "coords": (17.9500, 83.5000),
            "address": "Unnamed Rd, Dorathota, Bheemunipatnam, Anandapuram, Bhimili, Visakhapatnam, Andhra Pradesh, India",
            "image": "Jyothi_Hospital.jpeg"
        },
        "54": {
            "name": "Apollo 247 Diagnostics",
            "coords": (17.9000, 83.4500),
            "address": "Main Road, Bheemili, Bheemunipatnam, Near Bus Stand, Visakhapatnam, Andhra Pradesh 531163, India",
            "image": "Apollo_247.jpeg"
        },
        "55": {
            "name": "Indus Hospitals",
            "coords": (17.7055, 83.2985),
            "address": "KGH Down Road, Near Jagadamba Junction, Maharani Peta, Visakhapatnam, Andhra Pradesh 530002, India",
            "image": "Indus_Hospitals.jpeg"
        },
        "56": {
            "name": "Rhea Hospital (Metrox Multi-Speciality Hospital)",
            "coords": (17.7092, 83.3042),
            "address": "No. 18-01-03, K.G.H. Down, Maharanipeta, Jagadamba Junction, Visakhapatnam, Andhra Pradesh 530002, India",
            "image": "Rhea_Hospital.jpeg"
        },
        "57": {
            "name": "ICON Krishi Hospital (KIMS-ICON Hospital)",
            "coords": (17.6868, 83.1689),
            "address": "32-11-02, Sheela Nagar, BHPV Post, Visakhapatnam, Andhra Pradesh 530012, India",
            "image": "ICON_Krishi_Hospital.jpeg"
        },
        "58": {
            "name": "Government Hospital, Pedagantyada",
            "coords": (17.6833, 83.2170),
            "address": "Pedagantyada, Visakhapatnam, Andhra Pradesh 530044, India",
            "image": "Government_Hospital.jpeg"
        },
        "59": {
            "name": "Mahatma Gandhi Cancer Hospital",
            "coords": (17.7438204784948, 83.33582472438262),
            "address": "Plot No:1, Sector:7, Sector- 6, MVP Colony, Visakhapatnam, Andhra Pradesh 530017",
            "image": "Mahatma_Gandhi_Cancer.jpeg"
    },
        "60": {
            "name": "Area Hospital",
            "coords": (18.332635126670453, 82.884527548476),
            "address": "8VFP+W3G, Araku Valley, Andhra Pradesh 531149",
            "image": "AreaHos.jpg"
            },
        "61": {
            "name": "KEERTHANA HOSPITAL",
            "coords": (17.804464529322164, 83.35243139816284),
            "address": "Carshed, junction, 80 Feet Rd, Pothinamallayya Palem, Visakhapatnam, Andhra Pradesh 530041",
            "image": "KEERTHANA_HOSPITAL.jpg"
            },
        "62": {
            "name": "VIJAYASRI HOSPITAL",
            "coords": (17.79486218565601, 83.35346136640418),
            "address": "Opp Vizag Conventions, near YSR Vizag cricket stadium, Vasundhara Nagar, Pothinamallayya Palem, Visakhapatnam, Andhra Pradesh 530041",
            "image": "VIJAYASRI_HOSPITAL.jpg"
            },
        "62": {
            "name": "Vedanta Women and Children Hospital ",
            "coords": (17.804668828900574, 83.34779654099788),
            "address": "HIG 10-37, PM PALEM 80 FEET ROAD, BESIDES INDIAN BANK Madhurawada, Pothinamallayya Palem, Visakhapatnam, Andhra Pradesh 530041",
            "image": "vedanth.jpg"
            }
    }
}

GRAPH_CACHE_FILE = "visakhapatnam_graph.pkl"
place_name = "Visakhapatnam, Andhra Pradesh, India"

# Load or generate graph
if os.path.exists(GRAPH_CACHE_FILE):
    logger.info("Loading cached OSM graph...")
    with open(GRAPH_CACHE_FILE, 'rb') as f:
        G = pickle.load(f)
else:
    logger.info("Fetching OSM graph for Visakhapatnam...")
    G = ox.graph_from_place(place_name, network_type="drive", simplify=True)
    for u, v, data in G.edges(data=True):
        if 'length' in data and 'travel_time' not in data:
            data['travel_time'] = (data['length'] / 1000) / 40 * 3600
    with open(GRAPH_CACHE_FILE, 'wb') as f:
        pickle.dump(G, f)

def calculate_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    return sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2) * 111

@lru_cache(maxsize=None)
def get_nearest_node(coords):
    try:
        node = ox.distance.nearest_nodes(G, coords[1], coords[0], return_dist=False)
        if G.nodes[node].get("highway") or G.degree(node) > 0:
            return node
        return None
    except Exception as e:
        logger.error(f"Error in get_nearest_node: {e}")
        return None

# Synthetic travel time data generation and ML model training
def generate_synthetic_travel_data(n_samples=1000):
    np.random.seed(42)
    distances = np.random.uniform(1, 50, n_samples)
    vehicle_types = np.random.choice([0, 1, 2], n_samples)
    times_of_day = np.random.uniform(0, 24, n_samples)
    base_speeds = np.array([40, 30, 25])[vehicle_types]
    traffic_factor = 1 + 0.5 * np.sin(2 * np.pi * times_of_day / 24)
    travel_times = (distances / base_speeds) * 60 * traffic_factor
    return pd.DataFrame({
        'distance_km': distances,
        'vehicle_type': vehicle_types,
        'time_of_day': times_of_day,
        'travel_time': travel_times
    })

travel_data = generate_synthetic_travel_data()
X = travel_data[['distance_km', 'vehicle_type', 'time_of_day']]
y = travel_data['travel_time']
travel_time_model = LinearRegression()
travel_time_model.fit(X, y)
logger.info("ML model for travel time prediction trained.")

# Synthetic tourism popularity data generation and ML model training
def generate_synthetic_popularity_data():
    city_center = (17.726, 83.304)  # Dwaraka Nagar as center
    data = []
    for idx, place in visakhapatnam_data["tourism_places"].items():
        coords = place["coords"]
        dist_from_center = calculate_distance(coords, city_center)
        max_dist = max(calculate_distance(p["coords"], city_center) 
                      for p in visakhapatnam_data["tourism_places"].values())
        amenity_count = sum(1 for cat in ["accommodations", "restaurants", "hospitals"] 
                          for p in visakhapatnam_data[cat].values() 
                          if calculate_distance(coords, p["coords"]) <= 3)
        node = get_nearest_node(coords)
        accessibility = sum(1 for n in G.nodes 
                          if calculate_distance((G.nodes[n]["y"], G.nodes[n]["x"]), coords) <= 1) if node else 0
        popularity = 5 + 2 * (1 - dist_from_center / max_dist) + amenity_count * 0.5 + accessibility * 0.1
        popularity = np.clip(popularity, 0, 10)
        data.append({
            'dist_from_center': dist_from_center,
            'amenity_count': amenity_count,
            'accessibility': accessibility,
            'popularity': popularity
        })
    return pd.DataFrame(data)

popularity_data = generate_synthetic_popularity_data()
X_pop = popularity_data[['dist_from_center', 'amenity_count', 'accessibility']]
y_pop = popularity_data['popularity']
popularity_model = LinearRegression()
popularity_model.fit(X_pop, y_pop)
logger.info("ML model for tourism popularity trained.")

# Predict popularity for all tourism places
tourism_popularity_scores = {}
for idx, place in visakhapatnam_data["tourism_places"].items():
    coords = place["coords"]
    city_center = (17.726, 83.304)
    dist_from_center = calculate_distance(coords, city_center)
    amenity_count = sum(1 for cat in ["accommodations", "restaurants", "hospitals"] 
                       for p in visakhapatnam_data[cat].values() 
                       if calculate_distance(coords, p["coords"]) <= 3)
    node = get_nearest_node(coords)
    accessibility = sum(1 for n in G.nodes 
                       if calculate_distance((G.nodes[n]["y"], G.nodes[n]["x"]), coords) <= 1) if node else 0
    features = np.array([[dist_from_center, amenity_count, accessibility]])
    score = popularity_model.predict(features)[0]
    tourism_popularity_scores[idx] = np.clip(score, 0, 10)

nearest_nodes = {idx: get_nearest_node(data["coords"]) for idx, data in visakhapatnam_data["tourism_places"].items() if get_nearest_node(data["coords"]) is not None}

def coords_approx_equal(coord1: Tuple[float, float], coord2: Tuple[float, float], tolerance: float = 0.001) -> bool:
    return (abs(coord1[0] - coord2[0]) <= tolerance) and (abs(coord1[1] - coord2[1]) <= tolerance)

def geocode_location(location: str) -> Tuple[Tuple[float, float], str]:
    for category in ["general_locations", "tourism_places"]:
        for idx, data in visakhapatnam_data[category].items():
            if location.lower() == data["name"].lower():
                return data["coords"], None
    return None, f"Could not find location '{location}' in Visakhapatnam data."

def heuristic(node, goal):
    node_coords = (G.nodes[node]["y"], G.nodes[node]["x"])
    goal_coords = (G.nodes[goal]["y"], G.nodes[goal]["x"])
    return calculate_distance(node_coords, goal_coords)

def a_star_search(graph, start, goal):
    frontier = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    while frontier:
        current_f, current = heapq.heappop(frontier)
        if current == goal:
            break
        for neighbor in graph.neighbors(current):
            edge_data = graph[current][neighbor][0]
            tentative_g_score = g_score[current] + edge_data.get('travel_time', edge_data.get('length', 0))
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(frontier, (f_score[neighbor], neighbor))
    path = []
    current = goal
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

def predict_travel_time(nodes, vehicle_type, time_of_day=12.0):
    total_length = 0
    for i in range(len(nodes) - 1):
        u, v = nodes[i], nodes[i + 1]
        if G.has_edge(u, v):
            total_length += G[u][v][0].get('length', 0)
    distance_km = total_length / 1000
    vehicle_map = {"car": 0, "bike": 1, "auto": 2}
    features = np.array([[distance_km, vehicle_map.get(vehicle_type, 0), time_of_day]])
    return travel_time_model.predict(features)[0]

def get_continuous_tourism_route(start_node, end_node, tourism_nodes, max_distance_km=3):
    try:
        if not nx.has_path(G, start_node, end_node):
            logger.error("No path exists between start and end nodes")
            return [], []
        shortest_path = nx.shortest_path(G, start_node, end_node, weight="length")
        shortest_coords = [(G.nodes[node]["y"], G.nodes[node]["x"]) for node in shortest_path]
        nearby_tourism_nodes = []
        for idx, tourism_node in nearest_nodes.items():
            if tourism_node is None:
                continue
            tourism_coords = (G.nodes[tourism_node]["y"], G.nodes[tourism_node]["x"])
            min_dist_to_path = min(calculate_distance(tourism_coords, coord) for coord in shortest_coords)
            if min_dist_to_path <= max_distance_km:
                nearby_tourism_nodes.append((idx, tourism_node))
        if not nearby_tourism_nodes:
            logger.info("No tourism places within 3 km of the main route")
            return shortest_path, []
        def get_path_progress(node):
            node_coords = (G.nodes[node]["y"], G.nodes[node]["x"])
            distances = [calculate_distance(node_coords, coord) for coord in shortest_coords]
            closest_idx = distances.index(min(distances))
            return closest_idx
        nearby_tourism_nodes.sort(key=lambda x: get_path_progress(x[1]))
        route_nodes = [start_node]
        current_node = start_node
        visited_places = set()
        for idx, next_node in nearby_tourism_nodes:
            if next_node not in visited_places and nx.has_path(G, current_node, next_node):
                path_to_next = nx.shortest_path(G, current_node, next_node, weight="length")
                route_nodes.extend(path_to_next[1:])
                current_node = next_node
                visited_places.add(next_node)
        if current_node != end_node and nx.has_path(G, current_node, end_node):
            final_path = nx.shortest_path(G, current_node, end_node, weight="length")
            route_nodes.extend(final_path[1:])
        places_along_route = [idx for idx, node in nearby_tourism_nodes if node in visited_places]
        return route_nodes, places_along_route
    except Exception as e:
        logger.error(f"Error in get_continuous_tourism_route: {e}")
        return [], []

def fetch_places_near_location(location_coords: Tuple[float, float], category: str) -> List[Dict]:
    lat, lon = location_coords
    places = []
    local_category = {"accommodations": "accommodations", "restaurants": "restaurants", "hospitals": "hospitals"}[category]
    for idx, data in visakhapatnam_data[local_category].items():
        place_coords = data["coords"]
        dist = calculate_distance(location_coords, place_coords)
        if dist <= 3:
            image_path = f"{local_category}/{data['image']}"
            places.append({
                "name": data["name"],
                "coords": place_coords,
                "address": data.get("address", "Address not available"),
                "distance": dist,
                "rating": 5 - (dist / (3 / 5)),
                "image": image_path if os.path.exists(os.path.join(app.static_folder, 'images', image_path)) else "placeholder.jpg",
                "rec_idx": idx  # Add rec_idx for tracking
            })
    return sorted(places, key=lambda x: x["rating"], reverse=True)[:3]

@lru_cache(maxsize=100)
def get_sub_route(start_node, place_node):
    try:
        sub_route_nodes = nx.shortest_path(G, start_node, place_node, weight="length")
        return [(G.nodes[node]["y"], G.nodes[node]["x"]) for node in sub_route_nodes]
    except nx.NetworkXNoPath:
        return []

def create_map(user_coords, dest_coords, dest_name, route_coords, places_along_route, vehicle_type, 
              sub_routes=None, return_routes=None, recommendations=None, selected_recommendation=None, 
              map_filename="main_route_map.html", show_main_route=True):
    map_center = (17.6868, 83.2185)
    travel_map = folium.Map(location=map_center, zoom_start=12)
    
    # Start and destination markers (blue)
    folium.Marker(location=user_coords, popup="Starting Point", 
                 icon=folium.Icon(color="blue", icon="star")).add_to(travel_map)
    folium.Marker(location=dest_coords, popup=dest_name, 
                 icon=folium.Icon(color="blue", icon="star")).add_to(travel_map)
    
    # Tourism places with orange (popularity < 9.5) and red (popularity >= 9.5)
    for idx in places_along_route:
        data = visakhapatnam_data["tourism_places"][str(idx)]
        coords = data["coords"]
        name = data["name"]
        popularity = tourism_popularity_scores.get(idx, 0)
        color = "red" if popularity >= 9.5 else "orange"
        folium.Marker(
            location=coords,
            popup=f"{name} (Popularity: {popularity:.1f}/10)",
            icon=folium.Icon(color=color)
        ).add_to(travel_map)
    
    # Main route lines
    if show_main_route and route_coords:
        shortest_nodes = [ox.distance.nearest_nodes(G, coord[1], coord[0]) for coord in route_coords[0]]
        tourism_nodes = [ox.distance.nearest_nodes(G, coord[1], coord[0]) for coord in route_coords[1]]
        shortest_time = predict_travel_time(shortest_nodes, vehicle_type)
        tourism_time = predict_travel_time(tourism_nodes, vehicle_type)
        folium.PolyLine(locations=route_coords[0], color="blue", weight=2.5, opacity=1, 
                       popup=f"Shortest Route ({shortest_time:.1f} min via {vehicle_type})").add_to(travel_map)
        folium.PolyLine(locations=route_coords[1], color="green", weight=2.5, opacity=1, 
                       popup=f"Tourism Route ({tourism_time:.1f} min via {vehicle_type})").add_to(travel_map)
    
    # Sub-routes and return routes (purple)
    if sub_routes:
        for sub_route in sub_routes:
            folium.PolyLine(locations=sub_route, color="purple", weight=2, opacity=0.7, dash_array="5, 5").add_to(travel_map)
    if return_routes:
        for return_route in return_routes:
            folium.PolyLine(locations=return_route, color="purple", weight=2, opacity=0.7, dash_array="10, 10").add_to(travel_map)
    
    # Recommendations with distinct colors
    if recommendations and selected_recommendation:
        color = {"accommodations": "purple", "restaurants": "cadetblue", "hospitals": "green"}[selected_recommendation]
        for place in recommendations:
            coords = place["coords"]
            if coords[0] is None or coords[1] is None:
                continue
            popup = f"{place['name']}<br>{place['address']}<br>Distance: {place['distance']:.2f} km<br>Rating: {place['rating']:.1f}/5"
            folium.Marker(location=coords, popup=popup, icon=folium.Icon(color=color)).add_to(travel_map)
    
    map_path = os.path.join(MAPS_DIR, map_filename)
    travel_map.save(map_path)
    return map_filename

@app.route('/')
def index():
    locations = sorted([data["name"] for data in visakhapatnam_data["general_locations"].values()])
    return render_template('index.html', locations=locations)

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query', '').lower()
    locations = [data["name"] for data in visakhapatnam_data["general_locations"].values()]
    suggestions = [loc for loc in locations if query in loc.lower()]
    return jsonify(suggestions[:10])

@app.route('/plan_route', methods=['POST', 'GET'])
def plan_route():
    if request.method == 'POST':
        start_location = request.form['start_location'].strip()
        dest_location = request.form['dest_location'].strip()
        vehicle_type = request.form.get('vehicle_type', 'car')
    else:
        start_location = request.args.get('start_location')
        dest_location = request.args.get('dest_location')
        vehicle_type = request.args.get('vehicle_type', 'car')
        if not start_location or not dest_location:
            return render_template('index.html', error="Missing start or destination location.")
    
    cache_key = f"{start_location}_{dest_location}_{vehicle_type}"
    if request.method == 'GET' and cache_key in session:
        logger.info("Using cached route data")
        return render_template('result.html', **session[cache_key])
    
    user_coords, error = geocode_location(start_location)
    if user_coords is None:
        return jsonify({"error": error}) if request.method == 'POST' else render_template('index.html', error=error)
    lat, lon = user_coords
    if not (17.5 <= lat <= 18.5 and 82.5 <= lon <= 83.5):
        error_msg = "Starting location must be within Visakhapatnam (lat: 17.5-18.5, lon: 82.5-83.5)."
        return jsonify({"error": error_msg}) if request.method == 'POST' else render_template('index.html', error=error_msg)
    user_location_name = start_location.split(",")[0].strip()
    
    dest_coords, error = geocode_location(dest_location)
    if dest_coords is None:
        return jsonify({"error": error}) if request.method == 'POST' else render_template('index.html', error=error)
    lat, lon = dest_coords
    if not (17.5 <= lat <= 18.5 and 82.5 <= lon <= 83.5):
        error_msg = "Destination must be within Visakhapatnam (lat: 17.5-18.5, lon: 82.5-83.5)."
        return jsonify({"error": error_msg}) if request.method == 'POST' else render_template('index.html', error=error_msg)
    dest_name = dest_location.split(",")[0].strip()
    
    start_node = get_nearest_node(user_coords)
    end_node = get_nearest_node(dest_coords)
    if not start_node or not end_node:
        error_msg = "Could not find nearest nodes for route calculation."
        return jsonify({"error": error_msg}) if request.method == 'POST' else render_template('index.html', error=error_msg)
    
    shortest_nodes = a_star_search(G, start_node, end_node)
    shortest_coords = [(G.nodes[node]["y"], G.nodes[node]["x"]) for node in shortest_nodes]
    tourism_nodes = [node for idx, node in nearest_nodes.items() if node is not None]
    tourism_route_nodes, places_along_route = get_continuous_tourism_route(start_node, end_node, tourism_nodes, max_distance_km=3)
    tourism_coords = [(G.nodes[node]["y"], G.nodes[node]["x"]) for node in tourism_route_nodes] if tourism_route_nodes else shortest_coords
    route_coords = [shortest_coords, tourism_coords]
    
    map_filename = create_map(user_coords, dest_coords, dest_name, route_coords, places_along_route, vehicle_type)
    tourism_places = [{"idx": idx, "name": visakhapatnam_data["tourism_places"][idx]["name"]} for idx in places_along_route]
    route_coords_str = ";".join(f"{lat},{lon}" for lat, lon in shortest_coords)
    shortest_time = predict_travel_time(shortest_nodes, vehicle_type)
    tourism_time = predict_travel_time(tourism_route_nodes, vehicle_type) if tourism_route_nodes else shortest_time
    route_names = [user_location_name, dest_name]
    
    response = {
        "map_url": url_for('static', filename=f'maps/{map_filename}'),
        "tourism_places": tourism_places,
        "user_coords": user_coords,
        "dest_coords": dest_coords,
        "route_coords_str": route_coords_str,
        "places_along_route": places_along_route,
        "main_route_nodes": shortest_nodes,
        "start_node": start_node,
        "end_node": end_node,
        "destination": dest_name,
        "route": route_names,
        "start_location": start_location,
        "dest_location": dest_location,
        "vehicle_type": vehicle_type,
        "shortest_time": shortest_time,
        "tourism_time": tourism_time
    }
    session[cache_key] = response
    if request.method == 'POST':
        return jsonify(response)
    return render_template('result.html', **response)

@app.route('/tourism_place/<int:place_idx>')
def tourism_place(place_idx):
    user_coords = tuple(float(x) for x in request.args.get('user_coords').split(','))
    dest_coords = tuple(float(x) for x in request.args.get('dest_coords').split(','))
    route_coords_str = request.args.get('route_coords')
    main_route_nodes = [int(node) for node in request.args.get('main_route_nodes').split(',')]
    start_node = int(request.args.get('start_node'))
    end_node = int(request.args.get('end_node'))
    dest_name = request.args.get('dest_name')
    start_location = request.args.get('start_location')
    dest_location = request.args.get('dest_location')
    vehicle_type = request.args.get('vehicle_type', 'car')
    route_coords = [tuple(map(float, coord_pair.split(','))) for coord_pair in route_coords_str.split(';')]
    place_node = nearest_nodes.get(str(place_idx))
    sub_routes = []
    if start_node and place_node:
        sub_route_coords = get_sub_route(start_node, place_node)
        if sub_route_coords:
            sub_routes.append(sub_route_coords)
        else:
            return render_template('tourism_place.html', error="No sub-route found to this tourism place.")
    
    # Generate multiple images for the tourism place
    place_data = visakhapatnam_data["tourism_places"][str(place_idx)]
    base_image = place_data["image"]  # e.g., "rk_beach.jpg"
    base_name = os.path.splitext(base_image)[0]  # e.g., "rk_beach"
    place_images = [f"{base_name}_{i}.jpg" for i in range(1, 5)]  # Generate up to 4 images
    place_images = [img for img in place_images if os.path.exists(os.path.join(app.static_folder, 'images', img))]  # Filter existing images
    if not place_images:
        place_images = [base_image]  # Fallback to single image if no additional images exist

    map_filename = f"sub_route_{place_idx}_map.html"
    create_map(user_coords, dest_coords, dest_name, [], [str(place_idx)], vehicle_type, sub_routes=sub_routes, map_filename=map_filename, show_main_route=False)
    place_name = place_data["name"]
    return render_template('tourism_place.html', place_idx=place_idx, place_name=place_name, place_images=place_images,
                         map_url=url_for('static', filename=f'maps/{map_filename}'), user_coords=user_coords,
                         dest_coords=dest_coords, route_coords_str=route_coords_str, main_route_nodes=main_route_nodes,
                         start_node=start_node, end_node=end_node, dest_name=dest_name, start_location=start_location,
                         dest_location=dest_location, vehicle_type=vehicle_type)

@app.route('/get_recommendations/<int:place_idx>/<category>', methods=['GET'])
def get_recommendations(place_idx, category):
    user_coords = tuple(float(x) for x in request.args.get('user_coords').split(','))
    dest_coords = tuple(float(x) for x in request.args.get('dest_coords').split(','))
    route_coords_str = request.args.get('route_coords')
    main_route_nodes = [int(node) for node in request.args.get('main_route_nodes').split(',')]
    start_node = int(request.args.get('start_node'))
    end_node = int(request.args.get('end_node'))
    vehicle_type = request.args.get('vehicle_type', 'car')
    route_coords = [tuple(map(float, coord_pair.split(','))) for coord_pair in route_coords_str.split(';')]
    place_coords = visakhapatnam_data["tourism_places"][str(place_idx)]["coords"]
    recommendations = fetch_places_near_location(place_coords, category)
    map_filename = f"recommendation_{place_idx}_{category}_map.html"
    create_map(user_coords, dest_coords, request.args.get('dest_name'), [], [str(place_idx)], vehicle_type,
              recommendations=recommendations, selected_recommendation=category, map_filename=map_filename, show_main_route=False)
    return jsonify({"recommendations": recommendations, "map_url": url_for('static', filename=f'maps/{map_filename}')})

@app.route('/route_to_recommendation/<int:place_idx>/<category>/<rec_name>')
def route_to_recommendation(place_idx, category, rec_name):
    user_coords = tuple(float(x) for x in request.args.get('user_coords').split(','))
    dest_coords = tuple(float(x) for x in request.args.get('dest_coords').split(','))
    vehicle_type = request.args.get('vehicle_type', 'car')
    place_coords = visakhapatnam_data["tourism_places"][str(place_idx)]["coords"]
    place_node = get_nearest_node(place_coords)
    recommendations = fetch_places_near_location(place_coords, category)
    rec_data = next((rec for rec in recommendations if rec["name"] == rec_name), None)
    if not rec_data:
        return jsonify({"error": "Recommendation not found"}), 404
    rec_coords = rec_data["coords"]
    rec_node = get_nearest_node(rec_coords)
    sub_routes = []
    if place_node and rec_node:
        sub_route_to_rec = get_sub_route(place_node, rec_node)
        if sub_route_to_rec:
            sub_routes.append(sub_route_to_rec)
    map_filename = f"route_to_rec_{place_idx}_{category}_{rec_name.replace(' ', '_')}_map.html"
    create_map(user_coords, dest_coords, request.args.get('dest_name'), [], [str(place_idx)], vehicle_type,
              sub_routes=sub_routes, recommendations=[rec_data], selected_recommendation=category, map_filename=map_filename, show_main_route=False)
    return jsonify({"map_url": url_for('static', filename=f'maps/{map_filename}')})

if __name__ == "__main__":
    app.run(debug=True)