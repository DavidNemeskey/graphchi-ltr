## Computes the BM25 scores for mock_data.csv. Might not actually work now.
import sys

with open(sys.argv[1]) as inf:
    okapis = []
    for i, line in enumerate(inf):
        if i >= 1:
            fields = [float(f) if nof > 1 else f
                      for nof, f in enumerate(line.strip().split(','))]
            okapi = fields[3] * (fields[2] * 2.2 /
                    (fields[2] + 1.2 * (1 - 0.75 + 0.75 * fields[5] / 100)))
            okapis.append((okapi, (fields[0], fields[1])))
    print "\n".join("{0}, {1}, {2}".format(qd[0], qd[1], okapi)
            for okapi, qd in sorted(okapis, reverse=True))
