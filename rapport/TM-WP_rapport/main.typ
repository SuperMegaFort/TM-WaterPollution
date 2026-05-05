// main.typ
#import "template.typ": master_report

#show: master_report.with(
  title: "WaterWatcher : De l'inférence Deep Learning à l'application métier pour la surveillance de la pollution des rivières",
  author: "Cyril Telley",
  director: "M. Chanel",
  expert: "[Nom de l'Expert]",
  date: "Mai 2026",
)

#include "chapters/01-introduction.typ"
#include "chapters/02-evaluation.typ"
#include "chapters/03-methodologie.typ"
#include "chapters/04-resultats.typ"
#include "chapters/05-conclusion.typ"
