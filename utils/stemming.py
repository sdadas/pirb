import os
from typing import List
from rust_fst import Map
from joblib import load


POLISH_STOPWORDS = [
    "a", "aby", "ach", "acz", "aczkolwiek", "aj", "akurat", "albo", "ale", "ależ", "ani", "aż", "bardziej", "bardzo",
    "bez", "bo", "bowiem", "by", "byli", "bym", "bynajmniej", "być", "był", "była", "było", "były", "będzie", "będą",
    "cali", "cała", "cały", "chce", "chcę", "choć", "ci", "ciebie", "cię", "co", "cokolwiek", "coraz", "coś", "czasami",
    "czasem", "czemu", "czy", "czyli", "często", "daleko", "dla", "dlaczego", "dlatego", "do", "dobrze", "dokąd",
    "dość", "dr", "dużo", "dwa", "dwaj", "dwie", "dwoje", "dzieki", "dzisiaj", "dziś", "gdy", "gdyby", "gdyż", "gdzie",
    "gdziekolwiek", "gdzieś", "go", "godz", "hab", "i", "ich", "ii", "iii", "ile", "im", "inna", "inne", "inny",
    "innych", "inż", "iv", "ix", "iż", "ja", "jak", "jakaś", "jakby", "jaki", "jakichś", "jakie", "jakiś", "jakiż",
    "jakkolwiek", "jako", "jakoś", "je", "jeden", "jedna", "jednak", "jednakże", "jedno", "jednym", "jedynie", "jego",
    "jej", "jemu", "jest", "jestem", "jeszcze", "jeśli", "jeżeli", "już", "ją", "każdy", "kiedy", "kierunku", "kilka",
    "kilku", "kimś", "kto", "ktokolwiek", "ktoś", "która", "które", "którego", "której", "który", "których", "którym",
    "którzy", "ku", "lat", "lecz", "lub", "ma", "mają", "mam", "mamy", "mgr", "mi", "miał", "mimo", "między", "mnie",
    "mną", "mogą", "moi", "moim", "moja", "moje", "może", "możliwe", "można", "mu", "musi", "my", "mój", "na", "nad",
    "nam", "nami", "nas", "nasi", "nasz", "nasza", "nasze", "naszego", "naszych", "natomiast", "natychmiast", "nawet",
    "nic", "nich", "nie", "niego", "niej", "niemu", "nigdy", "nim", "nimi", "nią", "niż", "no", "nowe", "np", "nr",
    "o", "o.o.", "obok", "od", "ok", "około", "on", "ona", "one", "oni", "ono", "oraz", "owszem", "pan", "pana", "pani",
    "pl", "po", "pod", "podczas", "pomimo", "ponad", "ponieważ", "potem", "powinien", "powinna", "powinni", "powinno",
    "poza", "prawie", "prof", "przecież", "przed", "przede", "przedtem", "przez", "przy", "raz", "razie", "roku",
    "również", "sam", "sama", "się", "skąd", "sobie", "sobą", "sposób", "swoje", "są", "ta", "tak", "taka", "taki",
    "takich", "takie", "także", "tam", "te", "tego", "tej", "tel", "temu", "ten", "teraz", "też", "to", "tobie",
    "tobą", "toteż", "totobą", "trzeba", "trochę", "tu", "tutaj", "twoi", "twoim", "twoja", "twoje", "twym", "twój",
    "ty", "tych", "tylko", "tym", "tys", "tzw", "tę", "tęw", "u", "ul", "vi", "vii", "viii", "vol", "w", "wam", "wami",
    "was", "wasi", "wasz", "wasza", "wasze", "wcale", "we", "według", "wie", "wiele", "wielu", "więc", "więcej", "wraz",
    "wszyscy", "wszystkich", "wszystkie", "wszystkim", "wszystko", "wtedy", "www", "wy", "właśnie", "wśród", "xi",
    "xii", "xiii", "xiv", "xv", "z", "za", "zanim", "zapewne", "zawsze", "zaś", "zanim", "ze", "zeznowu", "znów",
    "został", "zresztą", "zł", "żaden", "żadna", "żadne", "żadnych", "że", "żeby"
]

ENGLISH_STOPWORDS = [
    "a", "able", "about", "above", "according", "accordingly", "across", "actually", "after", "afterwards", "again",
    "against", "ain't", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "an", "and", "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway",
    "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "aren't", "around", "as", "a's",
    "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below", "beside", "besides",
    "best", "better", "between", "beyond", "both", "brief", "but", "by", "came", "can", "cannot", "cant", "can't",
    "cause", "causes", "certain", "certainly", "changes", "clearly", "c'mon", "co", "com", "come", "comes",
    "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding",
    "could", "couldn't", "course", "c's", "currently", "definitely", "described", "despite", "did", "didn't",
    "different", "do", "does", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "during", "each",
    "edu", "eg", "eight", "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even",
    "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "far",
    "few", "fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly", "forth", "four",
    "from", "further", "furthermore", "get", "gets", "getting", "given", "gives", "go", "goes", "going", "gone", "got",
    "gotten", "greetings", "had", "hadn't", "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he",
    "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "here's", "hereupon",
    "hers", "herself", "he's", "hi", "him", "himself", "his", "hither", "hopefully", "how", "howbeit", "however",
    "how's", "i", "i'd", "ie", "if", "ignored", "i'll", "i'm", "immediate", "in", "inasmuch", "inc", "indeed",
    "insofar", "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "its", "it's", "itself", "i've",
    "just", "keep", "keeps", "kept", "know", "known", "knows", "last", "lately", "later", "latter", "latterly",
    "least", "less", "lest", "let", "let's", "like", "liked", "likely", "little", "look", "looking", "looks", "ltd",
    "mainly", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most",
    "mostly", "much", "must", "mustn't", "my", "myself", "name", "namely", "nd", "near", "nearly", "necessary",
    "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone",
    "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "obviously", "of", "off", "often", "oh", "ok",
    "okay", "old", "on", "once", "ones", "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours",
    "ourselves", "out", "outside", "over", "overall", "own", "particular", "particularly", "per", "perhaps", "placed",
    "please", "plus", "possible", "presumably", "probably", "provides", "que", "quite", "qv", "rather", "rd", "re",
    "really", "reasonably", "regarding", "regardless", "regards", "relatively", "respectively", "right", "s", "said",
    "same", "saw", "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming",
    "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "since", "six", "so", "some", "somebody",
    "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified",
    "specify", "specifying", "still", "sub", "such", "sup", "sure", "t", "take", "taken", "tell", "tends", "th",
    "than", "thank", "thanks", "thanx", "that", "thats", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "theres", "there's", "thereupon",
    "these", "they", "they'd", "they'll", "they're", "they've", "think", "third", "this", "thorough", "thoroughly",
    "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "took", "toward",
    "towards", "tried", "tries", "truly", "try", "trying", "t's", "twice", "two", "un", "under", "unfortunately",
    "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually",
    "value", "various", "very", "via", "viz", "vs", "want", "wants", "was", "wasn't", "way", "we", "we'd", "welcome",
    "well", "we'll", "went", "were", "we're", "weren't", "we've", "what", "whatever", "what's", "when", "whence",
    "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "where's", "whereupon", "wherever",
    "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "who's", "whose", "why", "why's",
    "will", "willing", "wish", "with", "within", "without", "wonder", "won't", "would", "wouldn't", "yes", "yet",
    "you", "you'd", "you'll", "your", "you're", "yours", "yourself", "yourselves", "you've"
]

GERMAN_STOPWORDS = [
    "ab", "aber", "alle", "allein", "allem", "allen", "aller", "allerdings", "allerlei", "alles", "allmählich",
    "allzu", "als", "alsbald", "also", "am", "an", "and", "ander", "andere", "anderem", "anderen", "anderer",
    "andererseits", "anderes", "anderm", "andern", "andernfalls", "anders", "anstatt", "auch", "auf", "aus",
    "ausgenommen", "ausser", "ausserdem", "außer", "außerdem", "außerhalb", "bald", "bei", "beide", "beiden",
    "beiderlei", "beides", "beim", "beinahe", "bereits", "besonders", "besser", "beträchtlich", "bevor", "bezüglich",
    "bin", "bis", "bisher", "bislang", "bist", "bloß", "bsp.", "bzw", "ca", "ca.", "content", "da", "dabei", "dadurch",
    "dafür", "dagegen", "daher", "dahin", "damals", "damit", "danach", "daneben", "dann", "daran", "darauf", "daraus",
    "darin", "darum", "darunter", "darüber", "darüberhinaus", "das", "dass", "dasselbe", "davon", "davor", "dazu",
    "daß", "dein", "deine", "deinem", "deinen", "deiner", "deines", "dem", "demnach", "demselben", "den", "denen",
    "denn", "dennoch", "denselben", "der", "derart", "derartig", "derem", "deren", "derer", "derjenige", "derjenigen",
    "derselbe", "derselben", "derzeit", "des", "deshalb", "desselben", "dessen", "desto", "deswegen", "dich", "die",
    "diejenige", "dies", "diese", "dieselbe", "dieselben", "diesem", "diesen", "dieser", "dieses", "diesseits", "dir",
    "direkt", "direkte", "direkten", "direkter", "doch", "dort", "dorther", "dorthin", "drauf", "drin", "drunter",
    "drüber", "du", "dunklen", "durch", "durchaus", "eben", "ebenfalls", "ebenso", "eher", "eigenen", "eigenes",
    "eigentlich", "ein", "eine", "einem", "einen", "einer", "einerseits", "eines", "einfach", "einführen", "einführte",
    "einführten", "eingesetzt", "einig", "einige", "einigem", "einigen", "einiger", "einigermaßen", "einiges", "einmal",
    "eins", "einseitig", "einseitige", "einseitigen", "einseitiger", "einst", "einstmals", "einzig", "entsprechend",
    "entweder", "er", "erst", "es", "etc", "etliche", "etwa", "etwas", "euch", "euer", "eure", "eurem", "euren",
    "eurer", "eures", "falls", "fast", "ferner", "folgende", "folgenden", "folgender", "folgendes", "folglich", "fuer",
    "für", "gab", "ganze", "ganzem", "ganzen", "ganzer", "ganzes", "gar", "gegen", "gemäss", "ggf", "gleich",
    "gleichwohl", "gleichzeitig", "glücklicherweise", "gänzlich", "hab", "habe", "haben", "haette", "hast", "hat",
    "hatte", "hatten", "hattest", "hattet", "heraus", "herein", "hier", "hier", "hinter", "hiermit", "hiesige", "hin",
    "hinein", "hinten", "hinter", "hinterher", "http", "hätt", "hätte", "hätten", "höchstens", "ich", "igitt", "ihm",
    "ihn", "ihnen", "ihr", "ihre", "ihrem", "ihren", "ihrer", "ihres", "im", "immer", "immerhin", "in", "indem",
    "indessen", "infolge", "innen", "innerhalb", "ins", "insofern", "inzwischen", "irgend", "irgendeine", "irgendwas",
    "irgendwen", "irgendwer", "irgendwie", "irgendwo", "ist", "ja", "je", "jed", "jede", "jedem", "jeden",
    "jedenfalls", "jeder", "jederlei", "jedes", "jedoch", "jemand", "jene", "jenem", "jenen", "jener", "jenes",
    "jenseits", "jetzt", "jährig", "jährige", "jährigen", "jähriges", "kam", "kann", "kannst", "kaum", "kein", "keine",
    "keinem", "keinen", "keiner", "keinerlei", "keines", "keineswegs", "klar", "klare", "klaren", "klares", "klein",
    "kleinen", "kleiner", "kleines", "koennen", "koennt", "koennte", "koennten", "komme", "kommen", "kommt", "konkret",
    "konkrete", "konkreten", "konkreter", "konkretes", "können", "könnt", "künftig", "leider", "machen", "man",
    "manche", "manchem", "manchen", "mancher", "mancherorts", "manches", "manchmal", "mehr", "mehrere", "mein",
    "meine", "meinem", "meinen", "meiner", "meines", "mich", "mir", "mit", "mithin", "muessen", "muesst", "muesste",
    "muss", "musst", "musste", "mussten", "muß", "mußt", "müssen", "müsste", "müssten", "müßt", "müßte", "nach",
    "nachdem", "nachher", "nachhinein", "nahm", "natürlich", "neben", "nebenan", "nehmen", "nein", "nicht", "nichts",
    "nie", "niemals", "niemand", "nirgends", "nirgendwo", "noch", "nun", "nur", "nächste", "nämlich", "nötigenfalls",
    "ob", "oben", "oberhalb", "obgleich", "obschon", "obwohl", "oder", "oft", "per", "plötzlich", "schließlich",
    "schon", "sehr", "sehrwohl", "seid", "sein", "seine", "seinem", "seinen", "seiner", "seines", "seit", "seitdem",
    "seither", "selber", "selbst", "sich", "sicher", "sicherlich", "sie", "sind", "so", "sobald", "sodass", "sodaß",
    "soeben", "sofern", "sofort", "sogar", "solange", "solch", "solche", "solchem", "solchen", "solcher", "solches",
    "soll", "sollen", "sollst", "sollt", "sollte", "sollten", "solltest", "somit", "sondern", "sonst", "sonstwo",
    "sooft", "soviel", "soweit", "sowie", "sowohl", "tatsächlich", "tatsächlichen", "tatsächlicher", "tatsächliches",
    "trotzdem", "ueber", "um", "umso", "unbedingt", "und", "unmöglich", "unmögliche", "unmöglichen", "unmöglicher",
    "uns", "unser", "unser", "unsere", "unsere", "unserem", "unseren", "unserer", "unseres", "unter", "usw", "viel",
    "viele", "vielen", "vieler", "vieles", "vielleicht", "vielmals", "vom", "von", "vor", "voran", "vorher", "vorüber",
    "völlig", "wann", "war", "waren", "warst", "warum", "was", "weder", "weil", "weiter", "weitere", "weiterem",
    "weiteren", "weiterer", "weiteres", "weiterhin", "weiß", "welche", "welchem", "welchen", "welcher", "welches",
    "wem", "wen", "wenig", "wenige", "weniger", "wenigstens", "wenn", "wenngleich", "wer", "werde", "werden", "werdet",
    "weshalb", "wessen", "wichtig", "wie", "wieder", "wieso", "wieviel", "wiewohl", "will", "willst", "wir", "wird",
    "wirklich", "wirst", "wo", "wodurch", "wogegen", "woher", "wohin", "wohingegen", "wohl", "wohlweislich", "womit",
    "woraufhin", "woraus", "worin", "wurde", "wurden", "während", "währenddessen", "wär", "wäre", "wären", "würde",
    "würden", "z.B.", "zB", "zahlreich", "zeitweise", "zu", "zudem", "zuerst", "zufolge", "zugleich", "zuletzt", "zum",
    "zumal", "zur", "zurück", "zusammen", "zuviel", "zwar", "zwischen", "ähnlich", "übel", "über", "überall",
    "überallhin", "überdies", "übermorgen", "übrig", "übrigens"
]

FRENCH_STOPWORDS = [
    "a", "abord", "absolument", "afin", "ah", "ai", "aie", "aient", "aies", "ailleurs", "ainsi", "ait", "allaient",
    "allo", "allons", "allô", "alors", "anterieur", "anterieure", "anterieures", "apres", "après", "as", "assez",
    "attendu", "au", "aucun", "aucune", "aucuns", "aujourd", "aujourd'hui", "aupres", "auquel", "aura", "aurai",
    "auraient", "aurais", "aurait", "auras", "aurez", "auriez", "aurions", "aurons", "auront", "aussi", "autant",
    "autre", "autrefois", "autrement", "autres", "autrui", "aux", "auxquelles", "auxquels", "avaient", "avais",
    "avait", "avant", "avec", "avez", "aviez", "avions", "avoir", "avons", "ayant", "ayez", "ayons", "b", "bah", "bas",
    "basee", "bat", "beau", "beaucoup", "bien", "bigre", "bon", "boum", "bravo", "brrr", "c", "car", "ce", "ceci",
    "cela", "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là", "celui", "celui-ci", "celui-là",
    "celà", "cent", "cependant", "certain", "certaine", "certaines", "certains", "certes", "ces", "cet", "cette",
    "ceux", "ceux-ci", "ceux-là", "chacun", "chacune", "chaque", "cher", "chers", "chez", "chiche", "chut", "chère",
    "chères", "ci", "cinq", "cinquantaine", "cinquante", "cinquantième", "cinquième", "clac", "clic", "combien",
    "comme", "comment", "comparable", "comparables", "compris", "concernant", "contre", "couic", "crac", "d", "da",
    "dans", "de", "debout", "dedans", "dehors", "deja", "delà", "depuis", "dernier", "derniere", "derriere",
    "derrière", "des", "desormais", "desquelles", "desquels", "dessous", "dessus", "deux", "deuxième", "deuxièmement",
    "devant", "devers", "devra", "devrait", "different", "differentes", "differents", "différent", "différente",
    "différentes", "différents", "dire", "directe", "directement", "dit", "dite", "dits", "divers", "diverse",
    "diverses", "dix", "dix-huit", "dix-neuf", "dix-sept", "dixième", "doit", "doivent", "donc", "dont", "dos", "douze",
    "douzième", "dring", "droite", "du", "duquel", "durant", "dès", "début", "désormais", "e", "effet", "egale",
    "egalement", "egales", "eh", "elle", "elle-même", "elles", "elles-mêmes", "en", "encore", "enfin", "entre",
    "envers", "environ", "es", "essai", "est", "et", "etant", "etc", "etre", "eu", "eue", "eues", "euh", "eurent",
    "eus", "eusse", "eussent", "eusses", "eussiez", "eussions", "eut", "eux", "eux-mêmes", "exactement", "excepté",
    "extenso", "exterieur", "eûmes", "eût", "eûtes", "f", "fais", "faisaient", "faisant", "fait", "faites", "façon",
    "feront", "fi", "flac", "floc", "fois", "font", "force", "furent", "fus", "fusse", "fussent", "fusses", "fussiez",
    "fussions", "fut", "fûmes", "fût", "fûtes", "g", "gens", "h", "ha", "haut", "hein", "hem", "hep", "hi", "ho",
    "holà", "hop", "hormis", "hors", "hou", "houp", "hue", "hui", "huit", "huitième", "hum", "hurrah", "hé", "hélas",
    "i", "ici", "il", "ils", "importe", "j", "je", "jusqu", "jusque", "juste", "k", "l", "la", "laisser", "laquelle",
    "las", "le", "lequel", "les", "lesquelles", "lesquels", "leur", "leurs", "longtemps", "lors", "lorsque", "lui",
    "lui-meme", "lui-même", "là", "lès", "m", "ma", "maint", "maintenant", "mais", "malgre", "malgré", "maximale",
    "me", "meme", "memes", "merci", "mes", "mien", "mienne", "miennes", "miens", "mille", "mince", "mine", "minimale",
    "moi", "moi-meme", "moi-même", "moindres", "moins", "mon", "mot", "moyennant", "multiple", "multiples", "même",
    "mêmes", "n", "na", "naturel", "naturelle", "naturelles", "ne", "neanmoins", "necessaire", "necessairement", "neuf",
    "neuvième", "ni", "nombreuses", "nombreux", "nommés", "non", "nos", "notamment", "notre", "nous", "nous-mêmes",
    "nouveau", "nouveaux", "nul", "néanmoins", "nôtre", "nôtres", "o", "oh", "ohé", "ollé", "olé", "on", "ont", "onze",
    "onzième", "ore", "ou", "ouf", "ouias", "oust", "ouste", "outre", "ouvert", "ouverte", "ouverts", "o|", "où", "p",
    "paf", "pan", "par", "parce", "parfois", "parle", "parlent", "parler", "parmi", "parole", "parseme", "partant",
    "particulier", "particulière", "particulièrement", "pas", "passé", "pendant", "pense", "permet", "personne",
    "personnes", "peu", "peut", "peuvent", "peux", "pff", "pfft", "pfut", "pif", "pire", "pièce", "plein", "plouf",
    "plupart", "plus", "plusieurs", "plutôt", "possessif", "possessifs", "possible", "possibles", "pouah", "pour",
    "pourquoi", "pourrais", "pourrait", "pouvait", "prealable", "precisement", "premier", "première", "premièrement",
    "pres", "probable", "probante", "procedant", "proche", "près", "psitt", "pu", "puis", "puisque", "pur", "pure",
    "q", "qu", "quand", "quant", "quant-à-soi", "quanta", "quarante", "quatorze", "quatre", "quatre-vingt", "quatrième",
    "quatrièmement", "que", "quel", "quelconque", "quelle", "quelles", "quelqu'un", "quelque", "quelques", "quels",
    "qui", "quiconque", "quinze", "quoi", "quoique", "r", "rare", "rarement", "rares", "relative", "relativement",
    "remarquable", "rend", "rendre", "restant", "reste", "restent", "restrictif", "retour", "revoici", "revoilà",
    "rien", "s", "sa", "sacrebleu", "sait", "sans", "sapristi", "sauf", "se", "sein", "seize", "selon", "semblable",
    "semblaient", "semble", "semblent", "sent", "sept", "septième", "sera", "serai", "seraient", "serais", "serait",
    "seras", "serez", "seriez", "serions", "serons", "seront", "ses", "seul", "seule", "seulement", "si", "sien",
    "sienne", "siennes", "siens", "sinon", "six", "sixième", "soi", "soi-même", "soient", "sois", "soit", "soixante",
    "sommes", "son", "sont", "sous", "souvent", "soyez", "soyons", "specifique", "specifiques", "speculatif", "stop",
    "strictement", "subtiles", "suffisant", "suffisante", "suffit", "suis", "suit", "suivant", "suivante", "suivantes",
    "suivants", "suivre", "sujet", "superpose", "sur", "surtout", "t", "ta", "tac", "tandis", "tant", "tardive", "te",
    "tel", "telle", "tellement", "telles", "tels", "tenant", "tend", "tenir", "tente", "tes", "tic", "tien", "tienne",
    "tiennes", "tiens", "toc", "toi", "toi-même", "ton", "touchant", "toujours", "tous", "tout", "toute", "toutefois",
    "toutes", "treize", "trente", "tres", "trois", "troisième", "troisièmement", "trop", "très", "tsoin", "tsouin",
    "tu", "té", "u", "un", "une", "unes", "uniformement", "unique", "uniques", "uns", "v", "va", "vais", "valeur",
    "vas", "vers", "via", "vif", "vifs", "vingt", "vivat", "vive", "vives", "vlan", "voici", "voie", "voient", "voilà",
    "voire", "vont", "vos", "votre", "vous", "vous-mêmes", "vu", "vé", "vôtre", "vôtres", "w", "x", "y", "z", "zut",
    "à", "â", "ça", "ès", "étaient", "étais", "était", "étant", "état", "étiez", "étions", "été", "étée", "étées",
    "étés", "êtes", "être", "ô"
]

STOPWORDS = {"pl": POLISH_STOPWORDS, "en": ENGLISH_STOPWORDS, "de": GERMAN_STOPWORDS, "fr": FRENCH_STOPWORDS}


class FSTLemmatizer:

    def __init__(self, data_path: str, lang: str):
        self.lang = lang
        self.lemmatizer = Map(path=f"{data_path}.fst")
        self.lemmas = load(f"{data_path}_lemmas.bin")

    def lemma(self, word: str):
        word = word.lower()
        if word in self.lemmatizer: return self.lemmas[self.lemmatizer[word]]
        else: return word

    def __call__(self, tokens: List[str]):
        return [self.lemma(token) for token in tokens]


def get_lemmatizer(lang: str) -> FSTLemmatizer:
    current_dir = os.path.dirname(os.path.realpath(__file__))
    resource_path = os.path.join(current_dir, "resources", f"{lang}_lemmatizer_lower")
    return FSTLemmatizer(resource_path, lang)


def get_stopwords(lang: str):
    return STOPWORDS[lang]
