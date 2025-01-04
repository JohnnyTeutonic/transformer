#include "../include/vocabulary.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>

Vocabulary::Vocabulary() {
  // Initialize special tokens first (guaranteed IDs)
  add_special_token("<pad>", 0);  // Padding token
  add_special_token("<unk>", 1);  // Unknown token
  add_special_token("<bos>", 2);  // Beginning of sequence
  add_special_token("<eos>", 3);  // End of sequence
  add_special_token("<mask>", 4); // Mask token for MLM

  // Store special token IDs
  pad_token_id = 0;
  unk_token_id = 1;
  bos_token_id = 2;
  eos_token_id = 3;

  initialize_basic_vocabulary();
}

void Vocabulary::add_word(const std::string &word) {
  if (token_to_id.find(word) == token_to_id.end()) {
    int id = id_to_token.size();
    token_to_id[word] = id;
    id_to_token.push_back(word);
  }
}

void Vocabulary::add_special_token(const std::string &token, int id) {
  token_to_id[token] = id;
  if (id >= id_to_token.size()) {
    id_to_token.resize(id + 1);
  }
  id_to_token[id] = token;
}

void Vocabulary::initialize_basic_vocabulary() {
  // Add articles and determiners first (before pronouns)
  std::vector<std::string> articles = {
      // Articles
      "a", "an", "the",
      
      // Demonstrative determiners
      "this", "that", "these", "those",
      
      // Possessive determiners
      "my", "your", "his", "her", "its", "our", "their",
      
      // Quantifiers and other determiners
      "all", "any", "both", "each", "every", "few", "many", "much",
      "several", "some", "such", "no", "none", "neither", "either",
      
      // Numbers as determiners
      "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
      "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
      
      // Other common determiners
      "another", "other", "what", "whatever", "which", "whichever",
      "whose", "enough", "various", "certain", "plenty", "lots", "most",
      "least", "last", "next", "previous", "same", "certain",
      
      // Distributive determiners
      "each", "every", "either", "neither"
  };

  // Basic pronouns and their contractions
  std::vector<std::string> pronouns = {
      "i",       "me",        "my",       "mine",     "myself",
      "you",     "your",      "yours",    "yourself", "he",
      "him",     "his",       "himself",  "she",      "her",
      "hers",    "herself",   "it",       "its",      "itself",
      "we",      "us",        "our",      "ours",     "ourselves",
      "they",    "them",      "their",    "theirs",   "themselves",
      "this",    "that",      "these",    "those",    "who",
      "whom",    "whose",     "which",    "what",     "whatever",
      "whoever", "whomever",  "anyone",   "everyone", "someone",
      "nobody",  "everybody", "somebody", "anyone",   "everyone",
      "no one",  "each",      "either",   "neither",  "many",
      "few",     "several",   "all",      "both",     "any",
      "some",    "oneself",   "y'all",    "youse",    "thee",
      "thou",    "thy",       "thine",    "ye",       "yon",
      "yonder",  "whichever", "whatsoever", "whosoever", "whomsoever"};

  // Common contractions and their variations
  std::vector<std::string> contractions = {
      "i'm",      "i've",     "i'll",     "i'd",     "you're",    "you've",
      "you'll",   "you'd",    "he's",     "he'll",   "he'd",      "she's",
      "she'll",   "she'd",    "it's",     "it'll",   "it'd",      "we're",
      "we've",    "we'll",    "we'd",     "they're", "they've",   "they'll",
      "they'd",   "isn't",    "aren't",   "wasn't",  "weren't",   "haven't",
      "hasn't",   "hadn't",   "doesn't",  "don't",   "didn't",    "won't",
      "wouldn't", "can't",    "couldn't", "mustn't", "shouldn't", "mightn't",
      "shan't",   "let's",    "that's",   "who's",   "what's",    "here's",
      "there's",  "where's",  "when's",   "why's",   "how's",     "daren't",
      "needn't",  "oughtn't", "ain't",    "y'all're", "y'all've", "y'all'll",
      "ma'am",    "o'clock",  "'tis",     "'twas",   "g'day",     "y'know",
      "d'you",    "c'mon",    "dunno",    "gonna",   "gotta",     "wanna",
      "gimme",    "lemme",    "kinda",    "sorta",   "hafta",     "oughta",
      "supposta", "useta",    "coulda",   "woulda",  "shoulda",   "musta"};

  // Common verbs with all their forms
  std::vector<std::string> verbs = {
      // Basic verbs
      "be", "am", "is", "are", "was", "were", "being", "been", "have", "has",
      "had", "having", "do", "does", "did", "doing", "done",
      // Common action verbs
      "go", "goes", "went", "going", "gone", "say", "says", "said", "saying",
      "get", "gets", "got", "getting", "gotten", "make", "makes", "made",
      "making", "know", "knows", "knew", "knowing", "known", "think", "thinks",
      "thought", "thinking", "take", "takes", "took", "taking", "taken", "see",
      "sees", "saw", "seeing", "seen", "come", "comes", "came", "coming",
      "want", "wants", "wanted", "wanting", "look", "looks", "looked",
      "looking", "use", "uses", "used", "using", "find", "finds", "found",
      "finding", "give", "gives", "gave", "giving", "given", "tell", "tells",
      "told", "telling", "work", "works", "worked", "working", "call", "calls",
      "called", "calling", "try", "tries", "tried", "trying", "ask", "asks",
      "asked", "asking", "need", "needs", "needed", "needing", "feel", "feels",
      "felt", "feeling", "become", "becomes", "became", "becoming", "leave",
      "leaves", "left", "leaving", "put", "puts", "putting", "mean", "means",
      "meant", "meaning", "keep", "keeps", "kept", "keeping", "let", "lets",
      "letting", "begin", "begins", "began", "beginning", "begun", "seem",
      "seems", "seemed", "seeming", "help", "helps", "helped", "helping",
      "talk", "talks", "talked", "talking", "turn", "turns", "turned",
      "turning", "show", "shows", "showed", "showing", "shown",
      // Additional verbs
      "write", "writes", "wrote", "writing", "written", "read", "reads", "reading",
      "sing", "sings", "sang", "singing", "sung", "dance", "dances", "danced",
      "dancing", "play", "plays", "played", "playing", "run", "runs", "ran",
      "running", "jump", "jumps", "jumped", "jumping", "swim", "swims", "swam",
      "swimming", "swum", "eat", "eats", "ate", "eating", "eaten", "drink",
      "drinks", "drank", "drinking", "drunk", "sleep", "sleeps", "slept",
      "sleeping", "walk", "walks", "walked", "walking", "fly", "flies", "flew",
      "flying", "flown", "draw", "draws", "drew", "drawing", "drawn",
      // Adding frequently occurring verbs from logs
      "prepare", "wait", "compete", "meet", "collaborate", "repair",
      "cook", "rush", "entertain", "hop", "code", "respond", "train",
      "examine", "soar", "maintain", "hunt", "patrol", "meditate",
      "consult", "study", "practice", "deploy", "serve", "rehearse",
      "build", "analyze", "learn", "drive", "create", "gather", "sit",
      "teach", "worship", "visit", "test", "clean", "operate", "mix",
      "treat", "research", "counsel", "fight", "glide", "preside",
      "rest", "settle", "pray", "organize", "file", "type", "experiment",
      "observe", "perform", "collect", "plan"
  };

  // Common prepositions and conjunctions
  std::vector<std::string> connectors = {
      // Prepositions
      "in", "on", "at", "to", "for", "with", "by", "from", "up", "about",
      "into", "over", "after", "beneath", "under", "above", "below", "behind",
      "between", "beyond", "during", "except", "through", "toward", "within",
      "without", "across", "along", "around", "before", "beside", "besides",
      "down", "inside", "near", "off", "since", "upon", "within", "throughout",
      // Additional prepositions
      "amid", "amidst", "among", "amongst", "atop", "barring", "concerning",
      "considering", "despite", "excluding", "following", "including", "minus",
      "notwithstanding", "opposite", "outside", "past", "per", "plus",
      "regarding", "round", "save", "unlike", "versus", "via", "worth",
      // Conjunctions
      "and", "but", "or", "nor", "for", "yet", "so", "because", "although",
      "unless", "since", "while", "where", "if", "then", "else", "therefore",
      "however", "moreover", "furthermore", "nevertheless", "meanwhile",
      "afterwards", "consequently", "otherwise", "instead", "whereas",
      // Additional conjunctions
      "accordingly", "additionally", "albeit", "besides", "hence", "likewise",
      "namely", "notwithstanding", "provided", "similarly", "thus", "wherefore",
      "wherever", "whenever", "whence", "whereby", "wherein", "whereupon"};

  // Common adjectives
  std::vector<std::string> adjectives = {
      "good",      "new",       "first",      "last",   "long",      "great",
      "little",    "own",       "other",      "old",    "right",     "big",
      "high",      "different", "small",      "large",  "next",      "early",
      "young",     "important", "few",        "public", "bad",       "same",
      "able",      "best",      "better",     "low",    "late",      "general",
      "specific",  "certain",   "free",       "full",   "special",   "easy",
      "clear",     "recent",    "final",      "main",   "sure",      "real",
      "available", "local",     "particular", "hard",   "major",     "current",
      "nice",      "happy",     "serious",    "ready",  "simple",    "possible",
      "whole",     "short",     "private",    "past",   "beautiful", "strong",
      "quick",     
      // Additional adjectives
      "amazing",   "awesome",   "brilliant",  "calm",   "clever",    "colorful",
      "creative",  "curious",   "delicate",   "eager",  "elegant",   "energetic",
      "enormous",  "excellent", "excited",    "famous", "fantastic", "fierce",
      "friendly",  "gentle",    "gorgeous",   "graceful", "handsome", "healthy",
      "helpful",   "honest",    "humble",     "hungry", "innocent",  "intelligent",
      "kind",      "lively",    "lovely",     "lucky",  "magical",   "mysterious",
      "natural",   "patient",   "peaceful",   "perfect", "pleasant", "polite",
      "powerful",  "proud",     "quiet",      "rare",   "reliable",  "rich",
      "scared",    "shy",       "silly",      "smart",  "smooth",    "soft",
      "sweet",     "talented",  "tiny",       "tough",  "unique",    "warm",
      "wise",      "wonderful", "worried",    "young"};

  // Common nouns
  std::vector<std::string> nouns = {
      // People and roles
      "person", "people", "family", "friend", "parent", "mother", "father",
      "child", "baby", "teacher", "student", "doctor", "worker", "artist",
      // Additional people and roles
      "accountant", "actor", "architect", "athlete", "author", "baker",
      "banker", "barber", "carpenter", "chef", "clerk", "coach", "dancer",
      "dentist", "designer", "director", "driver", "engineer", "farmer",
      "firefighter", "judge", "lawyer", "mechanic", "musician", "nurse",
      "painter", "pilot", "plumber", "poet", "police", "professor", "programmer",
      "reporter", "sailor", "scientist", "secretary", "singer", "soldier",
      "surgeon", "tailor", "therapist", "trainer", "translator", "veterinarian",
      "waiter", "writer",

      // Places
      "home", "house", "school", "office", "store", "hospital", "city",
      "country", "world", "room", "building", "street", "park", "garden",
      // Additional places
      "airport", "apartment", "arena", "bank", "beach", "bridge", "cafe",
      "castle", "cathedral", "church", "cinema", "clinic", "college", "court",
      "factory", "farm", "gallery", "gym", "harbor", "hotel", "island",
      "laboratory", "library", "mall", "market", "museum", "palace", "prison",
      "restaurant", "shop", "stadium", "station", "studio", "theater", "tower",
      "university", "village", "warehouse", "zoo",

      // Time
      "time", "day", "night", "morning", "evening", "week", "month", "year",
      "today", "tomorrow", "minute", "hour", "moment", "future", "past",
      // Additional time-related
      "afternoon", "age", "century", "dawn", "decade", "dusk", "era", "eternity",
      "history", "lifetime", "midnight", "millennium", "noon", "period", "present",
      "season", "spring", "summer", "autumn", "winter", "twilight", "weekend",
      "yesterday",

      // Nature
      "water", "air", "earth", "fire", "sun", "moon", "star", "sky", "tree",
      "flower", "grass", "river", "ocean", "mountain", "forest",
      // Additional nature
      "aurora", "avalanche", "beach", "breeze", "brook", "canyon", "cave",
      "cliff", "cloud", "coast", "coral", "crater", "desert", "dew", "dust",
      "earthquake", "eclipse", "fog", "frost", "galaxy", "geyser", "glacier",
      "hill", "hurricane", "iceberg", "island", "lake", "landscape", "meteor",
      "mist", "oasis", "planet", "rain", "rainbow", "reef", "sand", "sea",
      "snow", "storm", "stream", "sunrise", "sunset", "thunder", "tornado",
      "valley", "volcano", "wave", "wind",

      // Objects
      "book", "phone", "computer", "car", "door", "window", "table", "chair",
      "bed", "food", "money", "paper", "key", "screen", "picture",
      // Additional objects
      "alarm", "album", "anchor", "arrow", "badge", "bag", "ball", "basket",
      "battery", "bell", "blanket", "bottle", "bowl", "box", "bracelet",
      "brush", "bucket", "button", "camera", "candle", "card", "carpet",
      "clock", "coin", "compass", "crown", "cup", "curtain", "desk", "diary",
      "dictionary", "dish", "doll", "envelope", "fan", "flag", "flask",
      "fork", "frame", "glass", "glove", "hammer", "hat", "helmet", "knife",
      "lamp", "lock", "magazine", "map", "medal", "mirror", "needle",
      "newspaper", "notebook", "package", "paint", "pen", "pencil", "pillow",
      "plate", "radio", "ribbon", "ring", "rope", "ruler", "scissors", "shelf",
      "shoe", "soap", "spoon", "stamp", "stapler", "sword", "telescope",
      "ticket", "tool", "torch", "toy", "umbrella", "vase", "wallet", "watch",
      "wheel", "wire",

      // Abstract concepts
      "life", "death", "love", "hate", "peace", "war", "truth", "lie", "idea",
      "thought", "dream", "hope", "fear", "mind", "soul",
      // Additional abstract concepts
      "ability", "achievement", "action", "adventure", "advice", "age", "anger",
      "anxiety", "art", "balance", "beauty", "belief", "blame", "chance",
      "change", "chaos", "choice", "comfort", "communication", "confidence",
      "conflict", "confusion", "connection", "consciousness", "control",
      "courage", "creativity", "crisis", "culture", "curiosity", "democracy",
      "destiny", "difference", "difficulty", "dignity", "discipline",
      "discovery", "diversity", "doubt", "duty", "education", "emotion",
      "energy", "equality", "evil", "excellence", "existence", "experience",
      "failure", "faith", "fame", "fate", "freedom", "friendship", "fun",
      "future", "glory", "goal", "goodness", "grace", "gratitude", "grief",
      "growth", "guilt", "happiness", "harmony", "health", "heaven", "hell",
      "history", "honor", "humanity", "humor", "identity", "imagination",
      "independence", "infinity", "influence", "information", "innocence",
      "inspiration", "intelligence", "interest", "intuition", "irony",
      "joy", "justice", "kindness", "knowledge", "language", "laughter",
      "law", "liberty", "logic", "loneliness", "loss", "luck", "luxury",
      "magic", "meaning", "memory", "mercy", "miracle", "mystery", "nature",
      "necessity", "need", "opportunity", "pain", "passion", "patience",
      "perception", "perfection", "philosophy", "pleasure", "politics",
      "possibility", "poverty", "power", "pride", "principle", "progress",
      "promise", "prosperity", "purpose", "quality", "quantity", "question",
      "reality", "reason", "recognition", "religion", "respect",
      "responsibility", "revenge", "risk", "romance", "sacrifice", "safety",
      "satisfaction", "science", "security", "self", "sense", "serenity",
      "shame", "silence", "simplicity", "sin", "skill", "society", "solitude",
      "sorrow", "spirit", "strength", "stress", "structure", "success",
      "suffering", "surprise", "talent", "taste", "technology", "theory",
      "thinking", "time", "tolerance", "tradition", "trust", "understanding",
      "unity", "universe", "value", "victory", "violence", "virtue", "vision",
      "wealth", "wisdom", "wonder", "work", "world", "worth", "youth",

      // Adding frequently occurring nouns from logs
      "service", "simulator", "tunnel", "briefing", "hangar", "club",
      "hall", "space", "field", "players", "headquarters", "chamber",
      "workspace", "facility", "meat", "comedy", "bakery", "center",
      "tarmac", "auditorium", "bay", "ward", "pond", "wall", "ice",
      "lab", "sanctuary", "temple", "mosque", "observatory", "academy",
      "range", "shrine", "wine", "workshop", "chapel", "classroom",
      "base", "records", "pharmacy", "department", "pool", "galley",
      "rink", "track", "kitchen", "cellar", "precinct", "bar",
      "courtroom", "conference", "meeting", "district", "mat", "cargo",
      "anteroom", "monastery", "stage", "cockpit",

      // Teams and groups
      "crew", "teams", "assistants", "handlers", "technicians",
      
      // Facilities and rooms
      "examination", "consultation", "therapy", "radiology", "filing",

      // Adding missing professional and role-related nouns
      "aircraft", "baggage", "instructor", "instructors", "aviation", "mechanic",
      "mechanics", "controller", "controllers", "attendant", "attendants",
      "analyst", "analysts", "researcher", "researchers", "geologist", "geologists",
      "developer", "developers", "rescuer", "rescuers", "medic", "medics",
      "chemist", "chemists", "designer", "designers", "therapist", "therapists",
      "barista", "baristas", "paramedic", "paramedics", "climber", "climbers",
      "runner", "runners", "sous", "teacher", "teachers", "biologist", "biologists",
      "skater", "skaters", "pilgrim", "pilgrims", "attorney", "attorneys",
      "scientist", "scientists", "ranger", "rangers", "prosecutor", "prosecutors",
      "professor", "professors", "bartender", "bartenders", "worshipper", "worshippers",
      "priest", "priests", "pilot", "pilots", "scholar", "scholars", "pupil", "pupils",
      "firefighter", "firefighters", "inventor", "inventors", "officer", "officers",
      "musician", "musicians", "astronomer", "astronomers", "programmer", "programmers",
      "dancer", "dancers", "actor", "actors", "doctor", "doctors", "spectator",
      "spectators", "bowler", "bowlers", "clerk", "clerks", "tutor", "tutors",
      "baker", "bakers", "gamer", "gamers", "artist", "artists", "monk", "monks",
      "performer", "performers", "dj", "djs", "expert", "experts", "comedian",
      "comedians", "surgeon", "surgeons", "judge", "judges", "radiologist",
      "radiologists", "patient", "patients", "paralegal", "paralegals", "nurse",
      "nurses", "specialist", "specialists", "pharmacist", "pharmacists",
      "technician", "technicians", "boxer", "boxers", "police", "dentist",
      "dentists", "navigator", "navigators", "psychiatrist", "psychiatrists",
      "athlete", "athletes", "swimmer", "swimmers", "player", "players", "golfer",
      "golfers", "chef", "chefs", "sommelier", "sommeliers", "butcher", "butchers",
      "waiter", "waiters", "guard", "guards", "believer", "believers", "mediator",
      "mediators", "lawyer", "lawyers", "witness", "witnesses", "reporter",
      "reporters", "physicist", "physicists",

      // Adding missing animal-related nouns
      "duck", "ducks", "bird", "birds", "wolf", "wolves", "bear", "fish",
      "dog", "rabbit", "rabbits", "eagle", "eagles", "lion", "cat",

      // Adding missing place and facility-related nouns
      "seminary", "concert", "tournament", "alley", "patisserie",

      // Adding missing abstract/activity nouns
      "traffic", "shade", "lives", "prep", "testing", "driving", "bowling",
      "coffee",

      // Adding missing organizational/group nouns
      "congregation", "congregations", "audience", "audiences",

      // Adding missing specialized terms
      "air", "flight", "data", "defense", "ground", "legal", "operating",
      "dressing", "chambers"

      
  };

  // Adding frequently occurring nouns from training data
  std::vector<std::string> training_nouns = {
      "laboratory", "office", "library", "classroom", "auditorium", 
      "workshop", "facility", "center", "station", "hospital",
      "precinct", "headquarters", "base", "workspace", "studio",
      "gallery", "concert hall", "arena", "tournament", "club",
      "comedy club", "operating room", "clinic", "ward", "pharmacy",
      "dental office", "examination room", "therapy room", "consultation room",
      "radiology department", "clean room", "prep room", "server room",
      "data center", "control room", "dispatch center", "crime scene",
      "morgue", "courtroom", "conference room", "law office", "chambers",
      "meeting room", "anteroom", "filing room", "records room", "district office",
      "observatory", "testing chamber", "field station", "sample room",
      "computer lab", "wind tunnel", "briefing room", "simulator",
      "cargo hold", "service bay", "hangar", "tower", "cockpit",
      "galley", "tarmac", "terminal", "platform", "booth",
      "stage", "dressing room", "set", "production office", "trailer",
      "storage room", "catwalk", "pit", "basement", "garage",
      "sanctuary", "temple", "mosque", "monastery", "seminary",
      "shrine", "chapel", "church", "cathedral",
      "bakery", "kitchen", "wine cellar", "meat shop", "restaurant",
      "bar", "coffee shop", "patisserie", "dining room",
      "greenhouse", "conservatory", "field", "apiary", "vineyard",
      "ranch", "pasture", "woods", "site", "archive", "museum",
      "repository", "reading room", "bookstore", "register",
      "warehouse", "showroom", "mall", "building", "hallway",
      "desk", "cubicle", "boardroom", "department", "agency",
      "newsroom", "publishing house", "study", "home office",
      "garden", "production office", "wardrobe", "storage room",
      "booth", "music room", "practice room", "recital hall",
      "garage", "big top", "cage", "arena", "enclosure",
      "clinic", "salon", "stable", "barn", "smithy",
      "workshop", "basement", "scaffold", "roof", "window",
      "floor", "shop", "room", "runway", "spa", "studio",
      "class", "center", "office", "house", "apartment",
      "building", "basement", "control room", "call center",
      "dispatch center", "pole", "scene", "car", "crime scene",
      "range", "lab", "server room", "data center", "clean room",
      "fab", "facility", "factory", "line", "plant", "site",
      "workplace", "center", "building", "field", "plant",
      "control room", "neighborhood", "house", "station",
      "studio", "desk", "booth", "van", "suite", "dark room",
      "animation studio", "room", "set", "trailer", "stage",
      "storage room", "booth", "catwalk", "platform", "gym",
      "workshop", "studio", "booth", "music room", "practice room",
      "concert hall", "stage", "studio", "music room", "recital hall",
      "basement", "pit", "chapel", "church", "tower", "sidewalk",
      "subway", "stage", "circus", "tent", "carnival", "big top",
      "air", "cage", "arena", "enclosure", "clinic", "facility",
      "salon", "park", "house", "stable", "track", "barn",
      "smithy", "workshop", "basement", "building", "scaffold",
      "roof", "shop", "window", "floor", "shop", "studio",
      "office", "room", "workshop", "studio", "shop", "atelier",
      "studio", "runway", "salon", "shop", "nail salon", "spa",
      "salon", "studio", "gym", "class", "office", "clinic",
      "office", "center", "office", "agency", "house", "building",
      "apartment", "building", "basement", "desk", "control room",
      "call center", "dispatch center", "pole", "scene", "car",
      "crime scene", "range", "lab", "laboratory", "morgue",
      "lab", "scene", "lab", "room", "office", "server room",
      "data center", "office", "studio", "office", "workspace",
      "studio", "lab", "office", "desk", "office", "workspace",
      "room", "office", "department", "lab", "office", "laboratory",
      "workshop", "lab", "shop", "lab", "clean room", "facility",
      "factory", "line", "plant", "workplace", "site", "building",
      "roof", "tower", "plant", "control room", "center", "field",
      "poles", "neighborhood", "house", "roof", "station", "studio",
      "desk", "studio", "helicopter", "scene", "truck", "van",
      "control room", "office", "booth", "studio", "booth", "suite",
      "dark room", "facility", "studio", "office", "studio", "desk",
      "studio", "room", "studio", "set", "office", "studio", "set",
      "room", "office", "studio", "room", "setup", "station",
      "facility", "venue", "office", "venue", "location", "center",
      "gallery", "office", "space", "gallery", "auction house",
      "office", "shop", "showroom", "studio", "facility", "field",
      "wilderness", "habitat", "ocean", "ship", "bridge", "deck",
      "boat", "waters", "port", "pier", "docks", "cab", "warehouse",
      "office", "warehouse", "mailroom", "route", "street",
      "sorting facility", "road", "highway", "street", "city",
      "curb", "parking lot", "garage", "street", "sidewalk",
      "street", "site", "cab", "site", "site", "road", "span",
      "shaft", "mine", "field", "site", "office", "computer",
      "office", "studio", "office", "field", "office", "field",
      "lab", "station", "field", "lab", "field", "lab", "laboratory",
      "facility", "plant", "facility", "office", "workplace",
      "office", "site", "office", "department", "office", "office",
      "desk", "office", "call center", "office", "bank", "counter",
      "office", "trading floor", "desk", "office", "institution",
      "office", "department", "secure room", "facility", "office",
      "agency", "embassy", "consulate", "booth", "conference",
      "classroom", "school", "clinic", "office", "clinic",
      "surgery center", "office", "hospital", "center", "office",
      "hospital", "operating room", "ER", "clinic", "office",
      "clinic", "laboratory", "lab", "imaging room", "imaging center",
      "exam room", "clinic", "facility", "office", "hospital",
      "center", "drugstore", "shop", "office", "clinic", "office",
      "surgical center", "office", "clinic", "office", "dental lab",
      "laboratory"
  };

  // Add additional professional and role words
  std::vector<std::string> additional_roles = {
      "engineer", "manager", "operator", "worker", "designer", "driver",
      "monitor", "officer", "broadcaster", "dentist", "director", "researcher",
      "agent", "developer", "digger", "inspector", "instructor", "overseer",
      "planner", "trainer", "arranger", "auditor", "checker", "coordinator",
      "filmmaker", "installer", "leader", "maker", "master", "member",
      "performer", "player", "producer", "reporter", "teacher", "actor",
      "administrator", "animator", "announcer", "crafter", "curator", "dealer",
      "deliverer", "doctor", "editor", "evaluator", "firefighter", "gamer",
      "handler", "investigator", "medical", "organizer", "owner", "painter",
      "pathologist", "producer", "ranger", "reviewer", "shooter", "sketcher",
      "tester", "writer", "acrobat", "adjuster", "advertiser", "advisor",
      "aesthetician", "ambassador", "anchor", "anesthesiologist", "anthropologist",
      "appraiser", "archaeologist", "archivist", "assembler", "assessor",
      "assistant", "astronomer", "auctioneer", "audiologist", "auditioner",
      "baker", "banker", "barber", "bartender", "beekeeper", "believer",
      "biller", "bookkeeper", "bookseller", "botanist", "bowler", "boxer",
      "brainstormer", "breaker", "broker", "builder", "bus driver", "business analyst",
      "busker", "butcher", "cabinetmaker", "calculator", "captain", "cardiologist",
      "carpenter", "carrier", "cartographer", "cashier", "cataloger",
      "ceramicist", "character animator", "chauffeur", "cinematographer",
      "civil engineer", "claims processor", "climate scientist", "climber",
      "clown", "collector", "commentator", "composer", "conductor",
      "conservationist", "conservator", "consultant", "content creator",
      "controller", "coordinator", "copywriter", "coroner", "counselor",
      "creator", "cryptographer", "curator", "cutter", "cyber security",
      "dancer", "database administrator", "decorator", "deliverer", "detective",
      "devotee", "dialysis technician", "dietitian", "diplomat", "dispatcher",
      "dna analyst", "documenter", "drummer", "economist", "electrician",
      "electronics technician", "endodontist", "esl teacher", "event planner",
      "examiner", "excavator", "executive", "fabricator", "facilitator",
      "farmer", "farrier", "fashion designer", "fisherman", "fitness instructor",
      "fixer", "flavor chemist", "florist", "forecaster", "forensic scientist",
      "forester", "forklift operator", "furniture maker", "game developer",
      "gardener", "gis specialist", "glassblower", "glazier", "golfer",
      "graphics artist", "greeter", "grid operator", "groomer", "guitarist",
      "hairdresser", "hardware engineer", "historian", "host", "hr manager",
      "hydrologist", "hygienist", "illustrator", "industrial engineer",
      "influencer", "interior designer", "interpreter", "interviewer",
      "inventor", "inventory manager", "investigator", "investment banker",
      "janitor", "jeweler", "jockey", "journalist", "juggler", "keeper",
      "landlord", "landscaper", "lawyer", "learner", "librarian",
      "lighting technician", "lineman", "loader", "loan officer",
      "logistics manager", "longshoreman", "machine learning engineer",
      "magician", "mail carrier", "maintenance worker", "makeup artist",
      "manager", "manicurist", "marine biologist", "marketer",
      "mason", "massage therapist", "mathematician", "media specialist",
      "mediator", "meteorologist", "meter reader", "mobile developer",
      "modeler", "model", "molder", "motion graphics artist", "mri technician",
      "navigator", "network administrator", "neurologist", "novelist",
      "nuclear engineer", "nurse", "nutritionist", "obstetrician",
      "occupational therapist", "oceanographer", "oncologist", "ophthalmologist",
      "optician", "optimizer", "optometrist", "oral surgeon", "orchestra member",
      "organist", "orthodontist", "osha inspector", "paleontologist",
      "party planner", "pastry chef", "pattern maker", "paver",
      "payroll specialist", "pcb fabricator", "pediatrician", "periodontist",
      "personal trainer", "pet groomer", "phlebotomist", "photographer",
      "physical therapist", "physician", "pianist", "plumber", "podcaster",
      "post-production supervisor"
  };

  // Add raw word list
  std::vector<std::string> raw_words = {
      "engineers", "managers", "operators", "workers", "designers", "drivers", "officers",
      "dental", "directors", "manage", "researchers", "developers", "dig", "inspectors",
      "instructors", "oversee", "planners", "trainers", "arrange", "auditors", "film",
      "installers", "makers", "masters", "members", "performers", "players", "production",
      "reporters", "teachers", "actors", "administrators", "animators", "announce", "crane",
      "crime", "curators", "dealers", "doctors", "editors", "effects", "emergency",
      "environmental", "financial", "firefighters", "gamers", "handlers", "imaging",
      "insurance", "music", "organizers", "owners", "painters", "producers", "rangers",
      "server", "sound", "speech", "testers", "writers", "3d", "911", "adjusters",
      "advertisers", "advise", "advisors", "agricultural", "ai", "ambassadors", "anchors",
      "animal", "antique", "appraisers", "assembly", "assist", "astronomers", "auction",
      "auctioneers", "audio", "background", "bakers", "ballistics", "band", "bankers",
      "barbers", "bartenders", "beekeepers", "believers", "benefits", "billing", "blood",
      "bookkeepers", "booksellers", "bowlers", "boxers", "breakers", "brokers", "builders",
      "bulldozer", "bus", "business", "buskers", "butchers", "cabinet", "cable", "career",
      "carpenters", "carriers", "cartographers", "cashiers", "casting", "chain", "chambers",
      "character", "children", "choir", "cinematographers", "circuit", "civil", "claims",
      "climate", "climbers", "collections", "collectors", "commentators", "compliance",
      "compose", "composers", "concept", "conductors", "congregations", "conservators",
      "construction", "content", "controllers", "coordinators", "copywriters", "coroners",
      "costume", "counselors", "creators", "credit", "cryptographers", "customers", "cut",
      "cyber", "dam", "dancers", "dark", "darkroom", "database", "decide", "decorators",
      "delivery", "dialysis", "dispatchers", "display", "dna", "dressing", "drummers",
      "electronics", "elephant", "entrance", "esl", "esports", "estate", "event",
      "evidence", "exam", "examiners", "exhibition", "fabricators", "farmers", "farriers",
      "fashion", "fishermen", "fitness", "flavor", "forecasters", "forensic", "foresters",
      "forklift", "furniture", "game", "gardeners", "gis", "glassblowers", "glaziers",
      "golfers", "graders", "graphics", "grid", "groomers", "hairdressers", "hands",
      "hardware", "headquarters", "hold", "horse", "hot", "hr", "hydroelectric",
      "illustrators", "industrial", "influencers", "interior", "internet", "interns",
      "interpreters", "inventors", "inventory", "investigators", "investment", "jam",
      "janitors", "jewelers", "jugglers", "keepers", "landscapers", "lawyers", "learners",
      "lighting", "linemen", "loan", "logistics", "longshoremen", "loom", "machine",
      "mail", "maintenance", "makeup", "management", "marine", "marketers", "marketing",
      "massage", "media", "mediators", "meter", "mobile", "modelers", "motion", "mri",
      "muslims", "nail", "navigators", "network", "nuclear", "nurses", "occupational",
      "oceanographers", "optimize", "oral", "orchestra", "osha", "party", "pastry",
      "pattern", "paving", "payroll", "pcb", "personal", "pet", "photographers",
      "physical", "pick", "plumbers", "post", "post-production", "postal", "pottery",
      "practitioners", "preparers", "preservationists", "print", "printmakers", "process",
      "processors", "product", "professional", "professors", "program", "programmers",
      "project", "property", "props", "prosecutors", "prosthodontists", "protect",
      "publicists", "publishing", "qa", "race", "ranchers", "readers", "receptionists",
      "recital", "recruiters", "recycling", "regulators", "rescuers", "respiratory",
      "restore", "restorers", "rideshare", "riggers", "ringers", "ringmasters",
      "robotics", "roofers", "runners", "sailors", "salespeople", "sample", "satellite",
      "scouts", "screenwriters", "scrum", "sculptors", "seamstresses", "search", "secure",
      "sell", "semiconductor", "setters", "sew", "shape", "shepherds", "singers",
      "sitters", "skaters", "slide", "social", "software", "soil", "solar", "sommeliers",
      "sort", "sorting", "spectators", "sports", "statisticians", "stock", "stockers",
      "stone", "storage", "storyboard", "streamers", "structural", "stunt", "stylists",
      "superintendents", "supervise", "supervisors", "supply", "support", "surgery",
      "surgical", "surveillance", "surveyors", "sustainability", "swimmers", "swing",
      "system", "tailors", "tamers", "tax", "taxi", "technical", "telephone", "tellers",
      "tend", "tennis", "texture", "tile", "top", "toxicologists", "trade", "trading",
      "translators", "transport", "trapeze", "travel", "turbine", "tutors", "tv",
      "ultrasound", "university", "upholsterers", "urban", "utility", "ux", "valets",
      "video", "vintners", "violinists", "visual", "voice", "waiters", "walkers",
      "waste", "waters", "weather", "weavers", "web", "wedding", "welders", "wilderness",
      "wildlife", "witnesses", "worshippers", "x-ray", "yard", "yoga"
  };

  // Add all words to vocabulary
  std::vector<std::string> all_words;
  all_words.insert(all_words.end(), articles.begin(), articles.end());
  all_words.insert(all_words.end(), pronouns.begin(), pronouns.end());
  all_words.insert(all_words.end(), contractions.begin(), contractions.end());
  all_words.insert(all_words.end(), verbs.begin(), verbs.end());
  all_words.insert(all_words.end(), connectors.begin(), connectors.end());
  all_words.insert(all_words.end(), adjectives.begin(), adjectives.end());
  all_words.insert(all_words.end(), nouns.begin(), nouns.end());

  // Sort and remove duplicates
  std::sort(all_words.begin(), all_words.end());
  all_words.erase(std::unique(all_words.begin(), all_words.end()),
                  all_words.end());

  // Add each word
  for (const auto &word : all_words) {
    add_word(word);
  }
}

int Vocabulary::get_id(const std::string &token) const {
  auto it = token_to_id.find(token);
  return it != token_to_id.end() ? it->second : unk_token_id;
}

std::string Vocabulary::get_token(int id) const {
  return (id >= 0 && id < id_to_token.size()) ? id_to_token[id]
                                              : id_to_token[unk_token_id];
}

size_t Vocabulary::size() const { return id_to_token.size(); }

void Vocabulary::print_vocabulary_mappings() const {
  std::cout << "\n=== Special Tokens ===\n";
  std::cout << "PAD token: " << pad_token_id << " <-> "
            << id_to_token[pad_token_id] << "\n";
  std::cout << "UNK token: " << unk_token_id << " <-> "
            << id_to_token[unk_token_id] << "\n";
  std::cout << "BOS token: " << bos_token_id << " <-> "
            << id_to_token[bos_token_id] << "\n";
  std::cout << "EOS token: " << eos_token_id << " <-> "
            << id_to_token[eos_token_id] << "\n";

  std::cout << "\n=== Full Vocabulary ===\n";
  std::cout << "Total size: " << size() << " tokens\n\n";
}

bool Vocabulary::verify_mappings() const {
  bool valid = true;
  for (const auto &[token, id] : token_to_id) {
    if (id >= id_to_token.size()) {
      std::cout << "Error: Token '" << token << "' maps to invalid ID " << id
                << "\n";
      valid = false;
      continue;
    }
    if (id_to_token[id] != token) {
      std::cout << "Error: Inconsistent mapping for token '" << token << "'\n";
      std::cout << "token_to_id['" << token << "'] = " << id << "\n";
      std::cout << "id_to_token[" << id << "] = '" << id_to_token[id] << "'\n";
      valid = false;
    }
  }
  return valid;
}