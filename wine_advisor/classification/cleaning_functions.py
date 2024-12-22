import string
import re

def cleaning(sentence: str):
    """
    Cleaning de base du dataset Kaggle (majuscules, double-espaces, tirets, ponctuation)
    """

    if sentence is None:
        return None
    else:
        sentence = sentence.strip() ## On enleve les double espaces
        sentence = sentence.lower() ## On transforme tout en lowercase

        # Advanced cleaning
        sentence = sentence.replace('-',' ') ## On remplace les tirets par des espaces
        for punctuation in string.punctuation:
            sentence = sentence.replace(punctuation,'') ## On enleve la ponctuation

        return sentence


def remove_accents(raw_text: str):
    """
    Retrait des accents
    """

    if raw_text is None:
        return None
    else:
        raw_text = re.sub(u"[àáâãäå]", 'a', raw_text)
        raw_text = re.sub(u"[èéêë]", 'e', raw_text)
        raw_text = re.sub(u"[ìíîï]", 'i', raw_text)
        raw_text = re.sub(u"[òóôõö]", 'o', raw_text)
        raw_text = re.sub(u"[ùúûü]", 'u', raw_text)
        raw_text = re.sub(u"[ýÿ]", 'y', raw_text)
        raw_text = re.sub(u"[ß]", 'ss', raw_text)
        raw_text = re.sub(u"[ñ]", 'n', raw_text)

        return raw_text


def vintage_identification(text: str):
    """
    Extraction des millésime à partir du nom de la cuvée
    """

    vintage=''.join(char for char in text if char.isdigit())[:4]

    # On ne conserve que les millésimes depuis 1975 et on transforme en int
    if vintage=='':
        return 2023 # pour les effervescents notamment on fait l'hypothèse qu'ils ont 1 an
    else:
        return int(vintage)


def classification_type_of_wine(type: str):
    """
    Simplification du nombre de types de vins disponibles dans la base
    """

    dico = {
        'red':'rouge',
        'white':'blanc',
        'sparkling':'effervescent',
        'rose':'rose',
        'dessert':'blanc',
        'fortified':'fortifie',
        'portsherry':'fortifie'
    }

    if type in list(dico.keys()) :
        return str(dico[type])
    else:
        return "autres"


def classification_region(localisation: str):
    """
    Reecriture et normalisation des regions disponibles dans la base
    """

    dico = {
        'bordeaux':'bordeaux',
        'burgundy':'bourgogne',
        'rhone valley':'vallee du rhone',
        'loire valley':'vallee de la loire',
        'alsace':'alsace',
        'champagne':'champagne',
        'beaujolais':'beaujolais',
        'southwest france':'sud ouest',
        'provence':'provence',
        'languedocroussillon':'languedoc roussillon',
        'france other':'autres'
    }

    if localisation in list(dico.keys()) :
        return str(dico[localisation])
    else:
        return "autres"


def remove_key_word_winery(raw_text: str):
    """
    Simplification des noms des domaines en retirant les mots les plus courants
    """

    # On va retirer le top 20 mots les plus présents dans tout le dataset sur la colonne winery en France
    l = ['chateau', 'domaine', 'de', 'la', 'fils', 'et', 'du', 'des', 'les',
         'pere', 'saint', 'haut', 'maison', 'vignerons',
         'cave', 'le', 'domaines', 'vignobles', 'vignoble']

    x=raw_text.split(" ")
    raw_text2=' '.join(word for word in x if word not in l)

    return raw_text2


def classification_appellation(name: str):
    """
    Simplification du nombre d'appellations disponibles dans la base
    """

    # => NON UTILISE DANS LES FAITS
    # On ne garde le detail que du top 200 soit 96% des vins en France

    dico = {
        "alsace":"alsace",
        "champagne":"champagne",
        "cotes de provence":"cotes de provence",
        "sancerre":"sancerre",
        "bordeaux":"bordeaux",
        "chateauneuf du pape":"chateauneuf du pape",
        "chablis":"chablis",
        "bordeaux superieur":"bordeaux superieur",
        "cotes du rhone":"cotes du rhone",
        "cahors":"cahors",
        "saint emilion":"saint emilion",
        "vin de france":"vin de france",
        "bordeaux blanc":"bordeaux blanc",
        "blaye cotes de bordeaux":"cotes de blaye",
        "haut medoc":"haut medoc",
        "medoc":"medoc",
        "cremant dalsace":"cremant d alsace",
        "bourgogne":"bourgogne",
        "pouilly fuisse":"pouilly fuisse",
        "vin de pays doc":"vin de pays d oc",
        "beaujolais villages":"beaujolais villages",
        "muscadet sevre et maine":"muscadet sevre et maine",
        "bordeaux rose":"bordeaux",
        "graves":"graves",
        "morgon":"morgon",
        "gigondas":"gigondas",
        "cotes de gascogne":"cotes de gascogne",
        "beaune":"beaune",
        "costieres de nimes":"costieres de nimes",
        "pouilly fume":"pouilly fume",
        "pessac leognan":"pessac leognan",
        "moulin a vent":"moulin a vent",
        "meursault":"meursault",
        "vacqueyras":"vacqueyras",
        "vouvray":"vouvray",
        "coteaux daix en provence":"coteaux d aix en provence",
        "margaux":"margaux",
        "nuits st georges":"nuits st georges",
        "castillon cotes de bordeaux":"castillon cotes de bordeaux",
        "beaujolais":"beaujolais",
        "chassagne montrachet":"chassagne montrachet",
        "mercurey":"mercurey",
        "cotes du rhone villages":"cotes du rhone villages",
        "fleurie":"fleurie",
        "saint estephe":"saintestephe",
        "gevrey chambertin":"gevrey chambertin",
        "brouilly":"brouilly",
        "chinon":"chinon",
        "cadillac cotes de bordeaux":"cadillac cotes de bordeaux",
        "pays doc":"pays d oc",
        "cotes de bourg":"cotes de bourg",
        "puligny montrachet":"puligny montrachet",
        "pommard":"pommard",
        "saint veran":"saint veran",
        "pauillac":"pauillac",
        "cremant de bourgogne":"cremant de bourgogne",
        "savigny les beaune":"savigny les beaune",
        "macon villages":"macon villages",
        "touraine":"touraine",
        "pomerol":"pomerol",
        "volnay":"volnay",
        "santenay":"santenay",
        "lalande de pomerol":"lalande de pomerol",
        "menetou salon":"menetou salon",
        "saint julien":"saint julien",
        "gaillac":"gaillac",
        "chambolle musigny":"chambolle musigny",
        "lirac":"lirac",
        "tavel":"tavel",
        "julienas":"julienas",
        "coteaux varois en provence":"coteaux varois en provence",
        "bandol":"bandol",
        "vin mousseux":"autres",
        "vosne romanee":"vosne romanee",
        "ventoux":"ventoux",
        "madiran":"madiran",
        "cremant de loire":"cremant de loire",
        "val de loire":"val de loire",
        "condrieu":"condrieu",
        "mediterranee":"autres",
        "coteaux du languedoc":"coteaux du languedoc",
        "cotes de provence sainte victoire":"sainte victoire",
        "luberon":"luberon",
        "saint amour":"saint amour",
        "entre deux mers":"entre deux mers",
        "sauternes":"sauternes",
        "cotes de bordeaux":"cotes de bordeaux",
        "beaujolais blanc":"beaujolais blanc",
        "corbieres":"corbieres",
        "rully":"rully",
        "clos de vougeot":"clos de vougeot",
        "montagne saint emilion":"montagne saint emilion",
        "fronton":"fronton",
        "cote de brouilly":"cote de brouilly",
        "languedoc":"languedoc",
        "saint aubin":"saint aubin",
        "lussac saint emilion":"lussac saint emilion",
        "montagny":"montagny",
        "hermitage":"hermitage",
        "crozes hermitage":"crozes hermitage",
        "coteaux du languedoc":"coteaux du languedoc",
        "regnie":"regnie",
        "bergerac sec":"bergerac sec",
        "fronsac":"fronsac",
        "anjou":"anjou",
        "sauternes":"sauternes",
        "listrac medoc":"listrac medoc",
        "chiroubles":"chiroubles",
        "cremant de bordeaux":"cremant de bordeaux",
        "corse":"corse",
        "vire clesse":"vire clesse",
        "saint mont":"saint mont",
        "cotes de bordeaux":"cotes de bordeaux",
        "bourgogne hautes cotes de beaune":"hautes cotes de beaune",
        "pernand vergelesses":"pernand vergelesses",
        "minervois":"minervois",
        "saint nicolas de bourgueil":"saint nicolas de bourgueil",
        "clos de vougeot":"clos de vougeot",
        "chenas":"chenas",
        "comte tolosan":"comte tolosan",
        "saumur":"saumur",
        "petit chablis":"petit chablis",
        "corton charlemagne":"corton charlemagne",
        "coteaux du giennois":"coteaux du giennois",
        "givry":"givry",
        "corton":"corton",
        "muscadet cotes de grandlieu":"muscadet cotes de grandlieu",
        "puisseguin saint emilion":"puisseguin saint emilion",
        "aloxe corton":"aloxe corton",
        "hermitage":"hermitage",
        "bourgueil":"bourgueil",
        "cotes du tarn":"cotes du tarn",
        "vin de pays des cotes de gascogne":"vin de pays des cotes de gascogne",
        "cotes du lot":"cotes du lot",
        "rose danjou":"rose danjou",
        "arbois":"arbois",
        "saumur champigny":"saumur champigny",
        "ile de beaute":"ile de beaute",
        "buzet":"buzet",
        "savoie":"savoie",
        "cotes du marmandais":"cotes du marmandais",
        "quincy":"quincy",
        "cotes de provence la londe":"cotes de provence la londe",
        "cotes du roussillon":"cotes du roussillon",
        "cairanne":"cairanne",
        "cotes du roussillon villages":"cotes du roussillon villages",
        "chorey les beaune":"chorey les beaune",
        "bordeaux clairet":"bordeaux clairet",
        "reuilly":"reuilly",
        "francs cotes de bordeaux":"francs cotes de bordeaux",
        "cotes de bergerac":"cotes de bergerac",
        "savennieres":"savennieres",
        "morey saint denis":"morey saint denis",
        "beaujolais rose":"beaujolais rose",
        "echezeaux":"echezeaux",
        "beaujolais villages blanc":"beaujolais villages blanc",
        "pacherenc du vic bilh":"pacherenc du vic bilh",
        "marsannay":"marsannay",
        "macon lugny":"macon lugny",
        "beaumes de venise":"beaumes de venise",
        "monthelie":"monthelie",
        "saint joseph":"saint joseph",
        "france":"france",
        "vin de savoie":"vin de savoie",
        "cotes du ventoux":"cotes du ventoux",
        "faugeres":"faugeres",
        "fixin":"fixin",
        "macon fuisse":"macon fuisse",
        "muscat de beaumes de venise":"muscat de beaumes de venise",
        "grignan les adhemar":"grignan les adhemar",
        "bourgogne aligote":"bourgogne aligote",
        "coteaux bourguignons":"coteaux bourguignons",
        "saint chinian":"saint chinian",
        "ladoix":"ladoix",
        "cotes du luberon":"cotes du luberon",
        "cote rotie":"cote rotie",
        "pouilly loche":"pouilly loche",
        "saint peray":"saint peray",
        "cotes du jura":"cotes du jura",
        "cremant de limoux":"cremant de limoux",
        "premieres cotes de bordeaux":"premieres cotes de bordeaux",
        "limoux":"limoux",
        "auxey duresses":"auxey duresses",
        "cotes du rhone villages plan de dieu":"cotes du rhone villages plan de dieu",
        "cotes catalanes":"cotes catalanes",
        "patrimonio":"patrimonio",
        "macon chardonnay":"macon chardonnay",
        "blanquette de limoux":"blanquette de limoux",
        "cote de nuits villages":"cote de nuits villages",
        "saint romain":"saint romain",
        "charmes chambertin":"charmes chambertin",
        "cote chalonnaise":"cote chalonnaise",
        "monbazillac":"monbazillac",
        "premieres cotes de blaye":"premieres cotes de blaye",
        "macon la roche vineuse":"macon la roche vineuse",
        "cotes du rhone villages seguret":"cotes du rhone villages seguret",
        "coteaux varois":"coteaux varois",
        "coteaux du layon":"coteaux du layon",
        "clos de la roche":"clos de la roche"
    }

    if name in list(dico.keys()) :
        return str(dico[name])
    else:
        return "autres"
