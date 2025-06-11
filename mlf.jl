

struct noeuds
    noeuds::Vector{Int}
    arcs::Dict{Tuple{Int, Int}, Int}                    # clé: (noeud départ, noeud arrivée), val: temps de parcours
    motrices::Dict{Int, Int}                            # clé: motrice, val: noeud de départ
    wagons::Dict{Int, Int}                              # clé: wagon, val: noeud de départ
    planning::Dict{Int, Vector{Tuple{Int, Int}}}        # clé: wagon, val: (noeud d'arrivée, temps d'arrivée théorique)
    capacite::Int
    horizonTemp::Int
end

struct Solution
    trajetMotrices::Dict{Int, Vector{Tuple{Int, Int}}}      # clé: motrice, val: trajet (liste d'arcs parcourus)
    trajetWagons::Dict{Int, Vector{Tuple{Int, Int}}}        # clé: wagon: val: trajet (liste d'arcs parcourus)
    affectations::Dict{Int, Vector{Int}}                    # clé: wagon, val: liste des motrices d'affectations sucessives. Si l'aff. est <= 0, alors le wagon n'est affecté à aucune motrice.
end

function estBoucle(arc::Tuple{Int, Int})::Bool
    #=
    Retourne true si l'arc passé en argument est une boucle.
    =#
    return arc[1] == arc[2]
end

function verifierAdmissibilite(prob::noeuds, sol::Solution)::Bool
    #=
    Retourne true si la solution est admissible pour le problème passé en argument.
    =#
    # TODO: afficher des motifs d'inadmissibilité pour la solution
    horizonTemp = prob.horizonTemp


    # Cohérence de l'horizon temporel
    for trajet in values(sol.trajetMotrices)
        if length(trajet) != horizonTemp
            return false
        end
    end
    for trajet in values(sol.trajetWagons)
        if length(trajet) != horizonTemp
            return false
        end
    end
    for aff in values(sol.affectations)
        if length(aff) != horizonTemp
            return false
        end
    end

    # Respect de la structure du problème
    for (m, trajet) in sol.trajetMotrices
        noeudDepart = get(prob.motrices, m, -1)
        if !(m in keys(prob.motrices)) || (noeudDepart, noeudDepart) != trajet[1]
            return false
        end
    end
    for (w, trajet) in sol.trajetWagons
        noeudDepart = get(prob.wagons, w, -1)
        if !(w in keys(prob.wagons)) || (noeudDepart, noeudDepart) != trajet[1]
            return false
        end
    end

    # Immobilisation des wagons
    for t = 1:(horizonTemp-1)
        for (w, a) in sol.affectations     # a : vecteur contenant les affectations successives d'un wagon (identifiants de motrices).
            if a[t] <= 0 && sol.trajetWagons[w][t] != sol.trajetWagons[w][t+1]       # Si le wagon n'a pas d'affectation et change d'arc entre les instants t et t+1
                return false
            end
        end
    end

    # Affectation des wagons
    for t = 1:(horizonTemp-1)
        for (w, a) in sol.affectations
            if a[t] != a[t+1] && !estBoucle(sol.trajetWagons[w][t]) 
                return false
            end
        end
    end

    # Déplacement des wagons
    for t = 1:horizonTemp
        for (w, a) in sol.affectations
            m = a[t]
            if a[t] > 0 && sol.trajetWagons[w][t] != sol.trajetMotrices[m][t]
                return false
            end
        end
    end

    # Capacité des motrices
    for t = 1:horizonTemp
        affInstantT = Dict(w => a[t] for (w, a) in sol.affectations)        # Génère un nouveau dictionaire avec les affectations à l'instant t
        compteursMotrices = Dict{Int, Int}()                                # Compteur des occurences de chaque motrice
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
        if length(val) > 0 && maximum(val) > prob.capacite               # Si les affectations dépassent la capacité des motrices
            return false
        end
    end

    # Existence des arcs empruntés
    for t = 1:horizonTemp
        for m in keys(prob.motrices)
            for e in sol.trajetMotrices[m]
                if !(e in keys(prob.arcs))
                    return false
                end
            end
        end
    end

    # Continuité des parcours
    for t = 1:(horizonTemp-1)
        for m = keys(prob.motrices)
            e1 = sol.trajetMotrices[m][t]
            e2 = sol.trajetMotrices[m][t+1]
            if e1 != e2 && e1[2] != e2[1]           # Pour deux arcs successifs différents: si l'arc suivant ne commence pas à la fin de l'arc précédent, alors il y a discontinuité
                return false
            end
        end
    end

    # Exclusion des parcours sur un même arc
    for t = 1:horizonTemp
        for (m1, trajet1) in sol.trajetMotrices      # ve1/ve2 : vecteurs contenant les arcs successivement empruntés par les motrices m1 et m2     # TODO : utiliser des noms plus explicites
            for (m2, trajet2) in sol.trajetMotrices
                if m1 != m2 && trajet1[t] == trajet2[t] && !estBoucle(trajet1[t])       # Si deux motrices différentes se retrouvent sur un même arc reliant deux noeuds différents (qui n'est pas une boucle)
                    return false
                end
            end
        end
    end

    # Temps de parcours des arcs
    for t = 1:(horizonTemp-1)
        for m = keys(prob.motrices)
            pred = sol.trajetMotrices[m][t]
            suiv = sol.trajetMotrices[m][t+1]

            if pred == suiv || estBoucle(suiv)              # Si la motrice s'engage sur un nouvel arc qui n'est pas un arc de stationnement, alors on vérifie son temps de parcours 
                continue
            end

            d = prob.arcs[suiv]            # Temps de parcours sur l'arc "suiv"
            for i = 2:min(d, (horizonTemp-1)-t)                       # On vérifie que les arcs succédant à l'arc "suiv" respectent le temps de parcours
                if sol.trajetMotrices[m][t+i] != suiv
                    return false
                end
            end
        end
    end

    return true
end

function evaluerSolution(prob::noeuds, sol::Solution)::Int
    #=
    Calcule la valeur de la fonction objectif pour la solution passée en argument
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

    E = Dict{Tuple{Int, Int}, Int}()
    E[(1, 1)] = 1
    E[(2, 2)] = 1
    E[(3, 3)] = 1
    E[(1, 2)] = 2       # Temps de parcours plus long pour le test
    E[(2, 3)] = 1
    
    M = Dict{Int, Int}()
    M[1] = 1

    W = Dict{Int, Int}()
    W[1] = 2

    P = Dict{Int, Vector{Tuple{Int, Int}}}()        # TODO: simplifier la construction des éléments d'un problème
    P[1] = [(3, 5), (2, 6)]

    C = 1
    T = 6           # TODO: remplir automatiquement les vecteurs de la solution pour correspondre à l'horizon temporel (à faire dans une fonction spécifique)

    prob = noeuds(V, E, M, W, P, C, T)


    # Construction d'une solution
    EM = Dict{Int, Vector{Tuple{Int, Int}}}()
    EM[1] = [(1, 1), (1, 2), (1, 2), (2, 2), (2, 3), (3, 3)]

    EW = Dict{Int, Vector{Tuple{Int, Int}}}()
    EW[1] = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 3), (3, 3)]

    A = Dict{Int, Vector{Int}}()
    A[1] = [0, 0, 0, 1, 1, 1]

    sol = Solution(EM, EW, A)

    
    # Evaluation de l'admissibilité de la solution pour le problème
    println(verifierAdmissibilite(prob, sol))
    println(evaluerSolution(prob, sol))
end

main()