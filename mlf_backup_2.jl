
function recuitSimule(prob::Probleme; Tmax=100.0, Tmin=1.0, alpha=0.95, iter=100)
    planCourant = genererPlanAleatoire(prob)
    while !verifierPlan(prob, planCourant)
        planCourant = genererPlanAleatoire(prob)
    end
    coutCourant = evaluerPlan(prob, planCourant)
    meilleurPlan, meilleurCout = planCourant, coutCourant
    temp = Tmax

    while temp > Tmin
        for i in 1:iter
            planVoisin = deepcopy(planCourant)
            muterPlan!(planVoisin, prob)
            if !verifierPlan(prob, planVoisin)
                continue
            end
            coutVoisin = evaluerPlan(prob, planVoisin)
            Δ = coutVoisin - coutCourant

            if Δ < 0 || rand() < exp(-Δ / temp)
                planCourant = planVoisin
                coutCourant = coutVoisin
                if coutVoisin < meilleurCout
                    meilleurPlan, meilleurCout = deepcopy(planVoisin), coutVoisin
                end
            end
        end
        temp *= alpha
        println(temp)
    end

    return meilleurPlan, meilleurCout
end

function muterPlan!(plan::Plan, prob::Probleme)
    # TODO: corriger pour s'assurer que les mutations sont toutes valides. 
    horizon = prob.horizonTemp
    capacite = prob.capacite

    # Choix aléatoire d’une motrice et d’un temps de mutation
    m = rand(keys(plan.trajetMotrices))
    t = rand(2:horizon-1)

    # Si on est en milieu de parcours d’un arc long, on évite de couper
    arcCourant = plan.trajetMotrices[m][t]
    if arcCourant != plan.trajetMotrices[m][t-1]
        return  # évite mutation partielle
    end

    noeudActuel = arcCourant[2]
    candidats = [a for a in keys(prob.arcs) if a[1] == noeudActuel]

    # Exclusion des arcs utilisés par d’autres motrices au même instant (sauf boucles)
    for autre in keys(plan.trajetMotrices)
        if autre == m; continue; end
        if plan.trajetMotrices[autre][t] in candidats &&
           plan.trajetMotrices[autre][t][1] != plan.trajetMotrices[autre][t][2]
            deleteat!(candidats, findall(a -> a == plan.trajetMotrices[autre][t], candidats))
        end
    end

    if isempty(candidats); return; end
    newArc = rand(candidats)
    duree = prob.arcs[newArc]

    # Appliquer la mutation sur la motrice
    for dt in 0:min(duree-1, horizon - t)
        plan.trajetMotrices[m][t + dt] = newArc
    end

    # Recalculer les wagons affectés
    for w in keys(plan.trajetWagons)
        for dt in 0:min(duree-1, horizon - t)
            t2 = t + dt
            if plan.affectations[w][t2] == m
                plan.trajetWagons[w][t2] = newArc
            end
        end
    end
end