import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.
    """
    # Initialize the probability distribution dictionary
    prob_dist = dict()

    # Get the pages linked by the current page
    linked_pages = corpus[page]

    # Number of linked pages
    num_linked_pages = len(linked_pages)

    # If the current page has no outgoing links, assume it has links to all pages (including itself)
    if num_linked_pages == 0:
        linked_pages = corpus.keys()
        num_linked_pages = len(linked_pages)

    # Probability of choosing a link from the current page
    link_prob = damping_factor / num_linked_pages

    # Probability of choosing a page at random
    random_prob = (1 - damping_factor) / len(corpus)

    # Calculate the probability distribution
    for page in corpus:
        if page in linked_pages:
            prob_dist[page] = link_prob + random_prob
        else:
            prob_dist[page] = random_prob

    return prob_dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.
    """
    # Initialize the PageRank dictionary
    pagerank = dict()

    # Choose the first sample at random
    sample = random.choice(list(corpus.keys()))

    # Perform the sampling
    for _ in range(n):
        # Update the PageRank of the current sample
        pagerank[sample] = pagerank.get(sample, 0) + 1

        # Get the transition model for the current sample
        trans_model = transition_model(corpus, sample, damping_factor)

        # Choose the next sample based on the transition model
        sample = random.choices(list(trans_model.keys()), list(trans_model.values()), k=1)[0]

    # Normalize the PageRank values
    for page in pagerank:
        pagerank[page] /= n

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.
    """
    # Initialize the PageRank dictionary
    pagerank = {page: 1 / len(corpus) for page in corpus}

    # Iterate until convergence
    while True:
        # Copy the current PageRank values
        old_pagerank = pagerank.copy()

        # Update the PageRank of each page
        for page in pagerank:
            pagerank[page] = ((1 - damping_factor) / len(corpus)) + damping_factor * sum(old_pagerank[i] / len(corpus[i]) for i in corpus if page in corpus[i])

        # Check for convergence
        if all(abs(old_pagerank[page] - pagerank[page]) < 0.001 for page in pagerank):
            break

    return pagerank

if __name__ == "__main__":
    main()
