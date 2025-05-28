

struct Probleme
    #=
    V: noeuds du réseau.
    E: arcs du réseau. clé=description de l'arc, val=durée de parcours
    M: motrices. clé=identifiant de la motrice, val=noeud correspondant à la boucle de départ.
    W: wagons. clé=identifiant du wagon, val=noeud correspondant à la boucle de départ.
    P: planing du réseau. clé=identifiant du wagon, val1=noeud de destination, val2=borne min fenêtre temporelle, val3=borne max fenêtre temporelle.
    C: capacité des motrices.
    T: horizon temporel.
    =#
    V::Vector{Int}
    E::Dict{Tuple{Int, Int}, Int}
    M::Dict{Int, Int}
    W::Dict{Int, Int}
    P::Dict{Int, Tuple{Int, Int, Int}}
    C::Int
    T::Int
end

struct Solution
    #=
    EM: contient le chemin parcouru par chaque motrice. La clé est l'identifiant de la motrice, la valeur est la série d'arcs parcourus de t=1 à t=T.
    EW: contient le chemin parcouru par chaque wagon. La clé est l'identifiant de la motrice, la valeur est la série d'arcs parcourus de t=1 à t=T.
    A: contient les affectations successives des wagons aux motrices. La clé est l'identifiant du wagon, la valeur est la série d'identifiants des motrices de t=1 à t=T. 
    - Si une affectation est <= 0 à un instant donné, cela signifie que le wagon n'est affecté à aucune motrice à cet instant.
    =#
    EM::Dict{Int, Vector{Tuple{Int, Int}}}
    EW::Dict{Int, Vector{Tuple{Int, Int}}}
    A::Dict{Int, Vector{Int}}
end

function estBoucle(arc::Tuple{Int, Int})::Bool
    #=
    Retourne true si l'arc passé en argument est une boucle.
    =#
    return arc[1] == arc[2]
end

function verifierAdmissibilite(prob::Probleme, sol::Solution)::Bool
    #=
    Retourne true si la solution est admissible pour le problème passé en argument.
    =#
    # TODO: afficher des motifs d'inadmissibilité pour la solution
    T = prob.T


    # Cohérence de l'horizon temporel
    for (m, ve) in sol.EM
        if length(ve) != T
            return false
        end
    end
    for (w, ve) in sol.EW
        if length(ve) != T
            return false
        end
    end
    for (w, va) in sol.A
        if length(va) != T
            return false
        end
    end

    # Respect de la structure du problème
    for (m, ve) in sol.EM
        vd = get(prob.M, m, -1)     # vd: noeud correspondant à l'arc de départ
        if !(m in keys(prob.M)) || (vd, vd) != ve[1]
            return false
        end
    end
    for (w, ve) in sol.EW
        vd = get(prob.W, w, -1)     # vd: noeud correspondant à l'arc de départ
        if !(w in keys(prob.W)) || (vd, vd) != ve[1]
            return false
        end
    end

    # Immobilisation des wagons
    for t = 1:(T-1)
        for (w, a) in sol.A     # a : vecteur contenant les affectations successives d'un wagon (identifiants de motrices).
            if a[t] <= 0 && sol.EW[w][t] != sol.EW[w][t+1]       # Si le wagon n'a pas d'affectation et change d'arc entre les instants t et t+1
                return false
            end
        end
    end

    # Affectation des wagons
    for t = 1:(T-1)
        for (w, a) in sol.A
            if a[t] != a[t+1] && !estBoucle(sol.EW[w][t]) 
                return false
            end
        end
    end

    # Déplacement des wagons
    for t = 1:T
        for (w, a) in sol.A
            m = a[t]
            if a[t] > 0 && sol.EW[w][t] != sol.EM[m][t]
                return false
            end
        end
    end

    # Capacité des motrices
    for t = 1:T
        At = Dict(w => a[t] for (w, a) in sol.A)    # Génère un nouveau dictionaire avec les affectations à l'instant t
        cmpt = Dict{Int, Int}()                     # Compte le nombre d'occurrences de chaque motrice (clés de cmpt)
        for a in values(At)
            if a <= 0
                continue                            # Pas de décompte si le wagon n'a pas d'affectation
            end

            if a in keys(cmpt)
                cmpt[a] = cmpt[a] + 1
            else
                cmpt[a] = 1
            end
        end

        val = values(cmpt)
        if length(val) > 0 && maximum(val) > prob.C               # Si les affectations dépassent la capacité des motrices
            return false
        end
    end

    # Existence des arcs empruntés
    for t = 1:T
        for m in keys(prob.M)
            for e in sol.EM[m]
                if !(e in keys(prob.E))
                    return false
                end
            end
        end
    end

    # Continuité des parcours
    for t = 1:(T-1)
        for m = keys(prob.M)
            e1 = sol.EM[m][t]
            e2 = sol.EM[m][t+1]
            if e1 != e2 && e1[2] != e2[1]           # Pour deux arcs successifs différents: si l'arc suivant ne commence pas à la fin de l'arc précédent, alors il y a discontinuité
                return false
            end
        end
    end

    # Exclusion des parcours sur un même arc
    for t = 1:T
        for (m1, ve1) in sol.EM      # ve1/ve2 : vecteurs contenant les arcs successivement empruntés par les motrices m1 et m2
            for (m2, ve2) in sol.EM
                if m1 != m2 && ve1[t] == ve2[t] && !estBoucle(ve1[t])       # Si deux motrices différentes se retrouvent sur un même arc reliant deux noeuds différents (qui n'est pas une boucle)
                    return false
                end
            end
        end
    end

    # Temps de parcours des arcs
    for t = 1:(T-1)
        for m = keys(prob.M)
            pred = sol.EM[m][t]
            suiv = sol.EM[m][t+1]

            if pred == suiv || estBoucle(suiv)              # Si la motrice s'engage sur un nouvel arc qui n'est pas un arc de stationnement, alors on vérifie son temps de parcours 
                continue
            end

            d = prob.E[suiv]            # Temps de parcours sur l'arc "suiv"
            for i = 2:min(d, (T-1)-t)                       # On vérifie que les arcs succédant à l'arc "suiv" respectent le temps de parcours
                if sol.EM[m][t+i] != suiv
                    return false
                end
            end
        end
    end


    # Respect des commandes
    for (w, p) in prob.P
        atteint = false
        for t = 1:T
            if sol.EW[w][t] == (p[1], p[1])                 # Si le wagon est en stationnement sur son noeud de destination, alors la commande est remplie
                atteint = true
                break
            end
        end

        if !atteint
            return false
        end
    end

    return true
end

function evaluerSolution(prob::Probleme, sol::Solution)::Int
    #=
    Calcule la valeur de la fonction objectif pour la solution passée en argument
    TODO : permettre la réutilisation des wagons. Trier les commandes par wagon et par ordre chonologique (de tmin), et rattacher à chaque commande les arrivées dans le noeud de destination selon l'ordre de livraison.
    =#
    T = prob.T
    P = prob.P

    obj = 0
    for (w, p) in P
        dest = (p[1], p[1])     # Arc de destination
        tmin = p[2]
        tmax = p[3]

        for t=1:(T-1)
            pred = sol.EW[w][t]
            suiv = sol.EW[w][t+1]
            if pred != suiv && suiv == dest         # Si on arrive dans le noeud de destination, alors on calcul l'écart avec la fenêtre
                if t < tmin
                    obj = obj + (tmin - t)
                elseif t > tmax
                    obj = obj + (t - tmax)
                end
            end
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

    P = Dict{Int, Tuple{Int, Int, Int}}()
    P[1] = (3, 1, 3)

    C = 1
    T = 6           # TODO: remplir automatiquement les vecteurs de la solution pour correspondre à un horizon temporel (à faire dans une fonction spécifique)

    prob = Probleme(V, E, M, W, P, C, T)


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