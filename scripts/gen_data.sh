#! /bin/zsh

cmd="python synthetic_data.py"

for e (7); do
    print grid
    print $e;
    # eval ${cmd} -t grid -b 2 -e ${e};
done

for e ({6..12}); do
    print pl-tree
    print $e
    # eval ${cmd} -t pl-tree -b 2 -e ${e};
done

for e ({6..12}); do
    print balanced-tree
    print $e
    eval ${cmd} -t balanced-tree -b 2 -e ${e};
done

for e ({6..12}); do
    print er
    print $e
    eval ${cmd} -t er -b 2 -e ${e};
done

for e ({6..12}); do
    print barabasi
    print $e
    eval ${cmd} -t barabasi -b 2 -e ${e};
done

for e ({6..12}); do
    print kr-rand
    print $e    
    eval ${cmd} -t kr-rand -b 2 -e ${e};
done

for e ({6..12}); do
    print kr-peri
    print $e
    eval ${cmd} -t kr-peri -b 2  -e ${e};
done

for e ({6..12}); do
    print kr-hier
    print $e    
    eval ${cmd} -t kr-hier -b 2  -e ${e};
done
