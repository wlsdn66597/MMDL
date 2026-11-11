"""Training-only observations. Each stem uses the source photo with its authored field note."""
RECORDS = [
    ("12699","The field note places this site on the Pacific coast.",
     "Water passes between dark rocks and patches of green vegetation in the foreground.",
     [
       ("Which description combines the note with a visible feature?",["Coastal rocks beside water","A snowfield without exposed rock","Sand dunes with no water","Open grassland without rocks"],
        "Match the setting in the note with an object visible in the photograph.","The Pacific-coast note and the rocks beside water support the coastal-rock description."),
       ("Which claim would require observations on additional dates?",["This channel dries every summer.","There is water in the channel in this view.","There are rocks beside the channel.","Green vegetation is visible in front."],
        "A seasonal pattern requires observations across seasons.","The photograph shows water at one time; it cannot establish drying every summer.")
     ]),
    ("798","The animal in this photograph is a fire salamander.",
     "The animal has contrasting yellow bands and dark skin against a darker surface.",
     [
        ("Which description of its appearance is directly supported?",["Contrasting yellow and dark markings","A uniformly gray coat","Rows of white feathers","Only one uniform yellow shade"],
         "Use visible color boundaries on the animal.","Yellow markings contrast with the darker skin; the appearance is not uniform."),
        ("Which observation would require evidence beyond this single view?",["Every predator avoids these animals.","The skin has yellow markings.","The animal is on a dark surface.","The body also has dark markings."],
         "A behavioral claim about all predators requires observations of predators.","The markings are visible, but the picture shows no predator encounters that establish universal avoidance.")
     ]),
    ("11226","This photograph records a bald eagle approaching a surface.",
     "The bird extends its yellow feet and curved talons below its body.",
     [
       ("Which structure is visibly extended below the body?",["Feet bearing curved talons","A pair of antlers","A forked fish tail","Two membranous fins"],
        "Identify the visible structure at the location in the question.","The structures below the bird are its feet and curved talons."),
       ("Which outcome cannot be established from this photograph?",["The bird caught a fish seconds later.","Its feet are extended.","The bird has feathers.","Its curved talons are visible."],
        "A single frame does not show a later event.","The feet are extended, but the frame does not establish a later catch.")
     ]),
    ("14860","The specimen is identified as magnetite in its source caption.",
     "The photograph shows dark faces with bright reflected highlights on the specimen.",
     [
       ("Which observation can be checked in the photograph?",["Some faces reflect light brightly.","The specimen melts at room temperature.","It floated during a test.","It dissolves instantly in water."],
        "Choose a visible surface property rather than the outcome of a test.","Bright highlights on the dark faces directly support visible reflection."),
       ("Which claim would require a separate measurement?",["The mass is exactly 50 grams.","Some faces have bright highlights.","The visible specimen is mostly dark.","Its outline is irregular."],
        "An exact mass needs a scale or a reported measurement.","No scale or mass is shown, so a claim of exactly 50 grams needs a measurement.")
     ]),
    ("19705","The source caption identifies these growing plants as asparagus.",
     "Several upright green shoots emerge from the soil, with pointed tips above the surface.",
     [
       ("Which part of the scene can be located above ground?",["Pointed tips of upright shoots","A cluster of buried roots","A bed of unopened seeds underground","A network of underground tubers"],
        "Locate the part using the visible soil boundary.","The pointed shoot tips are visible above the soil surface."),
       ("Which conclusion would require measuring the plants again?",["Every shoot grows two centimeters per day.","Several shoots stand above the soil.","The visible shoots are green.","The tips point upward."],
        "A daily growth rate requires lengths measured at different times.","One image provides no second length or time interval for computing a daily rate.")
     ]),
    ("1645","The caption identifies the illustrated object as a ceramic plate.",
     "The illustration shows one intact plate with a decorated rim.",
     [
        ("Which statement is supported by the illustration?",["The rim has a decorative pattern.","The plate is shown broken in two.","A fork is resting on the plate.","The rim is a plain solid black band."],
         "Compare the visible rim with the descriptions.","The visible rim is decorated and the plate is shown intact."),
        ("Which statement would require an actual drop test?",["This plate survives a two-meter drop.","Its outline is oval in this view.","The rim has decoration.","The center appears light in color."],
         "Performance in a specified test must be observed or provided.","The illustration does not show a drop or its outcome.")
     ])
]
