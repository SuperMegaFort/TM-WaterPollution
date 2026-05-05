// template.typ
#let master_report(
  title: "",
  author: "",
  director: "",
  expert: "",
  date: "",
  body,
) = {
  set page(
    paper: "a4",
    margin: (left: 3cm, right: 2.5cm, top: 3cm, bottom: 3cm),
    numbering: "1",
    number-align: center,
  )

  set text(font: "New Computer Modern", size: 11pt, lang: "fr")
  set par(justify: true, leading: 0.65em)
  set heading(numbering: "1.1")

  // Page de garde (Inspirée du template HES-SO)
  align(center)[
    #v(2cm)
    #text(size: 16pt, weight: "bold")[MASTER OF SCIENCE IN ENGINEERING] \
    #text(size: 12pt)[Master of Science HES-SO in Engineering] \
    #v(1cm)
    #text(size: 12pt)[Orientation: Computer Science (CS)] \
    #v(3cm)
    #text(size: 20pt, weight: "bold")[#title] \
    #v(2cm)
    #text(size: 14pt)[Auteur : #author] \
    #v(1cm)
    #text(size: 12pt)[Sous la direction de : #director] \
    #text(size: 12pt)[Expert externe : #expert] \
    #v(1cm)
    #text(size: 12pt)[Lausanne, HES-SO Master, #date]
  ]

  pagebreak()

  // Abstract
  heading(numbering: none)[Abstract]
  [Insérer ici un résumé en anglais du projet WaterWatcher...]
  pagebreak()

  outline(title: "Table des matières", depth: 3)
  pagebreak()

  set page(numbering: "1")
  counter(page).update(1)
  body
}
