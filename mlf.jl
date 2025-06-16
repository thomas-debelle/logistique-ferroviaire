using DataStructures

const Arc = Tuple{Int, Int}
struct Probleme
    noeuds::Vector{Int}
    arcs::Dict{Arc, Int}                    # clé: (noeud départ, noeud arrivée), val: temps de parcours
    motrices::Dict{Int, Int}                # clé: motrice, val: noeud de départ
    wagons::Dict{Int, Int}                  # clé: wagon, val: noeud de départ
    planning::Dict{Int, Vector{Arc}}        # clé: wagon, val: (noeud d'arrivée, temps d'arrivée théorique)
    capacite::Int
    horizonTemp::Int
end

abstract type Mouvement end

struct Attendre <: Mouvement
    noeud::Int
    duree::Int
end

struct Atteler <: Mouvement
    wagon::Int
end

struct Deposer <: Mouvement
    wagon::Int
    noeud::Int
end

# Solution à un problème. Une solution détaille l'ensemble des mouvements successifs d'une motrice, qui peuvent être de plusieurs types.
const Solution = Dict{Int, Vector{Mouvement}}                     # clé: motrice, val: liste des mouvements de la motrice. 

const Trajets = Dict{Int, Vector{Arc}}       
const Affectations = Dict{Int, Vector{Int}}                     # clé: wagon, val: liste des motrices d'affectations sucessives. Si l'aff. est <= 0, alors le wagon n'est affecté à aucune motrice.
struct Plan
    #=
    Plan détaillé du parcours des motrices, et du parcours et des affectations des wagons.
    Un plan peut être généré à partir d'une solution en utilisant un algorithme de pathfinding pour compléter les mouvements dans l'ordre.
    =#
    trajetMotrices::Trajets                                     # clé: motrice, val: trajet (liste d'arcs parcourus)
    trajetWagons::Trajets                                       # clé: wagons, val: trajet (liste d'arcs parcourus)
    affectations::Affectations                    
end

function estBoucle(arc::Arc)::Bool
    #=
    Retourne true si l'arc passé en argument est une boucle.
    =#
    return arc[1] == arc[2]
end

function dijkstra(probleme::Probleme, depart::Int, arrivee::Int)
    # Initialisation des temps de parcours et des prédécesseurs
    distances = Dict{Int, Int}()
    predecesseurs = Dict{Int, Int}()
    for noeud in probleme.noeuds
        distances[noeud] = typemax(Int)
    end
    distances[depart] = 0

    # File de priorité (min-heap) pour stocker les noeuds à explorer
    pq = PriorityQueue{Int, Int}()
    enqueue!(pq, depart => 0)

    while !isempty(pq)
        
        noeud, distanceActuelle = dequeue_pair!(pq)             # Extraction du noeud avec la plus petite distance
        if distanceActuelle > distances[noeud]                  # Si on a déjà trouvé un chemin plus court vers ce noeud, ignorer
            continue
        end

        # Si le noeud actuel est le noeud d'arrivée, on a trouvé le chemin le plus court
        if noeud == arrivee
            chemin = Int[]
            noeudActuel = arrivee
            while haskey(predecesseurs, noeudActuel)
                pushfirst!(chemin, noeudActuel)
                noeudActuel = predecesseurs[noeudActuel]
            end
            pushfirst!(chemin, depart)                          # Ajout du noeud de départ
            return distances[arrivee], chemin
        end

        # Parcourir les voisins du noeud actuel
        for (arc, poidsArc) in probleme.arcs
            if arc[1] == noeud
                v = arc[2]
                nouvelleDistance = distanceActuelle + poidsArc

                # Si un chemin plus court est trouvé vers v
                if nouvelleDistance < distances[v]
                    distances[v] = nouvelleDistance
                    predecesseurs[v] = noeud
                    enqueue!(pq, v => nouvelleDistance)
                end
            end
        end
    end

    # Si le noeud d'arrivée n'est pas atteignable
    return -1, Int[]
end

function construireTrajet(prob::Probleme, depart::Int, arrivee::Int)::Vector
    #=
    Retourne le trajet développé entre deux noeuds d'un problème.
    =#
    cout, chemin = dijkstra(prob, depart, arrivee)
    trajet = []
    for i=1:(size(chemin,1)-1)
        arc = (chemin[i], chemin[i+1])
        dureeParcours = prob.arcs[arc]
        for d in 1:dureeParcours
            push!(trajet, arc)
        end
    end
    return trajet
end

function construirePlan(prob::Probleme, sol::Solution)::Plan
    #=
    Construit un plan à partir d'une solution.
    =#
    trajetMotrices = Trajets()
    trajetWagons = Trajets()

    # Initialisation des trajets à partir des conditions initiales du problème
    for (m, noeudDepart) in prob.motrices
        trajetMotrices[m] = [(noeudDepart, noeudDepart)]
    end

    for (w, noeudDepart) in prob.wagons
        trajetWagons[w] = [(noeudDepart, noeudDepart)]
    end

    # Réalisation des mouvements
    # TODO: voir comment gérer les éventuels conflits pour emprunter la même voie
    for (m, mouvements) in sol
        for mouvement in mouvements
            if mouvement isa Attendre
                dest = mouvement.noeud
                trajet = construireTrajet(prob, last(trajetMotrices[m])[1], dest)
                append!(trajetMotrices[m], trajet)              # Ajout du trajet jusqu'au noeud d'attente
                for d in 1:mouvement.duree
                    push!(trajetMotrices[m], (dest, dest))      # Ajout de l'attente
                end
            elseif mouvement isa Atteler
                # TODO: chercher l'emplacement du wagon 
                # Comment gérer les conflits d'attelage (si le wagon est déjà attelé à une autre motrice) -> ignorer simplement ?
                # Calculer le temps pour se rendre jusqu'au noeud de destination, et voir quelle est l'affectation du wagon à cet instant.
                # Réaliser le mouvement dans tous les cas, même si conflit d'attelage ?
            elseif mouvement isa Deposer

            end
        end
    end
end

function verifierPlan(prob::Probleme, plan::Plan)::Bool
    #=
    Retourne true si le plan est admissible pour le problème passé en argument.
    =#
    horizonTemp = prob.horizonTemp


    # Cohérence de l'horizon temporel
    for trajet in values(plan.trajetMotrices)
        if length(trajet) != horizonTemp
            @info "Les trajets des motrices ne correspondent pas à l'horizon temporel du problème."
            return false
        end
    end
    for trajet in values(plan.trajetWagons)
        if length(trajet) != horizonTemp
            @info "Les trajets des wagons ne correspondent pas à l'horizon temporel du problème."
            return false
        end
    end
    for aff in values(plan.affectations)
        if length(aff) != horizonTemp
            @info "Les affectations ne correspondent pas à l'horizon temporel du problème."
            return false
        end
    end

    # Respect de la structure du problème
    for (m, trajet) in plan.trajetMotrices
        noeudDepart = get(prob.motrices, m, -1)
        if !(m in keys(prob.motrices)) || (noeudDepart, noeudDepart) != trajet[1]
            @info "Les trajets des motrices ne respectent pas la structure du problème."
            return false
        end
    end
    for (w, trajet) in plan.trajetWagons
        noeudDepart = get(prob.wagons, w, -1)
        if !(w in keys(prob.wagons)) || (noeudDepart, noeudDepart) != trajet[1]
            @info "Les trajets des wagons ne respectent pas la structure du problème."
            return false
        end
    end

    # Immobilisation des wagons
    for t = 1:(horizonTemp-1)
        for (w, a) in plan.affectations     # a : vecteur contenant les affectations successives d'un wagon (identifiants de motrices).
            if a[t] <= 0 && plan.trajetWagons[w][t] != plan.trajetWagons[w][t+1]       # Si le wagon n'a pas d'affectation et change d'arc entre les instants t et t+1
                @info "Un wagon a changé d'arc alors qu'il n'été affecté à aucune motrice."
                return false
            end
        end
    end

    # Affectation des wagons
    for t = 1:(horizonTemp-1)
        for (w, a) in plan.affectations
            if a[t] != a[t+1] && !estBoucle(plan.trajetWagons[w][t]) 
                @info "Une affectation/désaffectation a été réalisée hors d'un arc de stationnement."
                return false
            end
        end
    end

    # Déplacement des wagons
    for t = 1:horizonTemp
        for (w, a) in plan.affectations
            m = a[t]
            if a[t] > 0 && plan.trajetWagons[w][t] != plan.trajetMotrices[m][t]
                @info "Un wagon ne s'est pas déplacé avec sa motrice d'affectation."
                return false
            end
        end
    end

    # Capacité des motrices
    for t = 1:horizonTemp
        affInstantT = Dict(w => a[t] for (w, a) in plan.affectations)       # Nouveau dictionaire avec les affectations à l'instant t
        compteursMotrices = Dict{Int, Int}()                                # Compteur d'occurences pour chaque motrice
        for a in values(affInstantT)
            if a <= 0
                continue                                                    # Pas de décompte si le wagon n'a pas d'affectation
            end

            if a in keys(compteursMotrices)
                compteursMotrices[a] = compteursMotrices[a] + 1
            else
                compteursMotrices[a] = 1
            end
        end

        val = values(compteursMotrices)
        if length(val) > 0 && maximum(val) > prob.capacite                  # Si les affectations dépassent la capacité des motrices
            @info "La capacité d'une motrice n'a pas été respectée."
            return false
        end
    end

    # Existence des arcs empruntés
    for t = 1:horizonTemp
        for m in keys(prob.motrices)
            for e in plan.trajetMotrices[m]
                if !(e in keys(prob.arcs))
                    @info "Un motrice a emprunté un arc qui n'existe pas."
                    return false
                end
            end
        end
    end

    # Continuité des parcours
    for t = 1:(horizonTemp-1)
        for m = keys(prob.motrices)
            e1 = plan.trajetMotrices[m][t]
            e2 = plan.trajetMotrices[m][t+1]
            if e1 != e2 && e1[2] != e2[1]           # Pour deux arcs successifs différents: si l'arc suivant ne commence pas à la fin de l'arc précédent, alors il y a discontinuité
                @info "Une rupture dans la continuité du parcours a été détectée"
                return false
            end
        end
    end

    # Exclusion des parcours sur un même arc
    for t = 1:horizonTemp
        for (m1, trajet1) in plan.trajetMotrices
            for (m2, trajet2) in plan.trajetMotrices
                if m1 != m2 && trajet1[t] == trajet2[t] && !estBoucle(trajet1[t])       # Si deux motrices différentes se retrouvent sur un même arc reliant deux noeuds différents (qui n'est pas une boucle)
                    @info "Deux motrices parcourent un même arc entre deux noeuds en même temps."
                    return false
                end
            end
        end
    end

    # Temps de parcours des arcs
    for t = 1:(horizonTemp-1)
        for m = keys(prob.motrices)
            pred = plan.trajetMotrices[m][t]
            suiv = plan.trajetMotrices[m][t+1]

            if pred == suiv || estBoucle(suiv)              # Si la motrice s'engage sur un nouvel arc qui n'est pas un arc de stationnement, alors on vérifie son temps de parcours 
                continue
            end

            d = prob.arcs[suiv]            # Temps de parcours sur l'arc "suiv"
            for i = 2:min(d, (horizonTemp-1)-t)                       # On vérifie que les arcs succédant à l'arc "suiv" respectent le temps de parcours
                if plan.trajetMotrices[m][t+i] != suiv
                    @info "Le temps de parcours d'un arc n'a pas été respecté."
                    return false
                end
            end
        end
    end

    return true
end

function evaluerPlan(prob::Probleme, sol::Plan)::Int
    #=
    Calcule la valeur de la fonction objectif pour le plan passé en argument.
    TODO: ajouter une variante permettant de considérer t comme un délai maximum et non un horaire prévisionnel.
    =#
    horizonTemp = prob.horizonTemp
    planningTheorique = prob.planning


    # Tri les étapes du planning chronologiquement selon le temps d'arrivée théorique
    for (w, val) in planningTheorique
        planningTheorique[w] = sort(val, by = x -> x[2])
    end

    # Construction du planning réel
    planningReel = deepcopy(planningTheorique)          # Construction initiale du planning réel, puis ajustement des temps d'étapes
    for (w, etapes) in planningReel
        for i in eachindex(etapes)
            noeudDest = etapes[i][1]
            etapes[i] = (noeudDest, typemax(Int))             # Par défaut, on considère que chaque étape n'est jamais complétée (instant d'arrivée à l'infini)
        end
        
        indiceEtape = 1
        nouvelleEtape = true
        t = 1
        
        while t <= horizonTemp && indiceEtape <= size(etapes, 1)
            if nouvelleEtape            # Début d'une étape
                noeudDest = etapes[indiceEtape][1]
                arcDest = (noeudDest, noeudDest)
                nouvelleEtape = false
            end

            if sol.trajetWagons[w][t] == arcDest  # Fin d'une étape
                planningReel[w][indiceEtape] = (noeudDest, t)
                indiceEtape = indiceEtape + 1
                nouvelleEtape = true
            end
            t = t + 1
        end
    end

    # Calcul de la fonction objectif à partir des écarts entre le planning théorique et le planning réel
    obj = 0
    for (w, etapes) in planningReel
        for pi in eachindex(etapes)
            ecart = abs(planningReel[w][pi][2] - planningTheorique[w][pi][2])
            obj = obj + ecart
        end
    end

    return obj
end


function main()
    # Construction d'un problème
    V = [1, 2, 3]

    E = Dict{Arc, Int}()
    E[(1, 1)] = 1
    E[(2, 2)] = 1
    E[(3, 3)] = 1
    E[(1, 2)] = 2       # Temps de parcours plus long pour le test
    E[(2, 3)] = 1
    
    M = Dict{Int, Int}()
    M[1] = 1

    W = Dict{Int, Int}()
    W[1] = 2

    P = Dict{Int, Vector{Arc}}()
    P[1] = [(3, 5), (2, 6)]

    C = 1
    T = 6           # TODO: remplir automatiquement les vecteurs de la solution pour correspondre à l'horizon temporel (à faire dans une fonction spécifique)

    prob = Probleme(V, E, M, W, P, C, T)

    # Construction d'un plan
    EM = Dict{Int, Vector{Arc}}()
    EM[1] = [(1, 1), (1, 2), (1, 2), (2, 2), (2, 3), (3, 3)]

    EW = Dict{Int, Vector{Arc}}()
    EW[1] = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 3), (3, 3)]

    A = Dict{Int, Vector{Int}}()
    A[1] = [0, 0, 0, 1, 1, 1]

    plan = Plan(EM, EW, A)

    # Construction d'un plan à partir d'une solution
    sol = Solution()
    sol[1] = [Attendre(2, 1)]
    construirePlan(prob, sol)

    
    # Evaluation de l'admissibilité du plan pour le problème
    println(verifierPlan(prob, plan))
    println(evaluerPlan(prob, plan))
end

main()