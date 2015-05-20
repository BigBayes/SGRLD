Stochastic Gradient Riemannian Langevin Dynamics for Latent Dirichlet Allocation

Sam Patterson and Yee Whye Teh
spatterson@gatsby.ucl.ac.uk and y.w.teh@stats.ox.ac.uk

(C) Copyright 2013, Sam Patterson and Yee Whye Teh

This is free software, you can redistribute it and/or modify it under
the terms of the GNU General Public License.

The GNU General Public License does not permit this software to be
redistributed in proprietary programs.

This software is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
USA

------------------------------------------------------------------------

This Python code implements the stochastic gradient Riemannian Langevin
dynamics (SGRLD) algorithm presented in the paper "Stochastic Gradient
Riemannian Langevin Dynamics on the Probability Simplex" by Sam Patterson
and Yee Whye Teh at NIPS 2013.  BibTeX entry:

@inproceedings{PatTeh2013a,
  Author = {S. Patterson and Y. W. Teh},
  Booktitle = {Advances in Neural Information Processing Systems},
  Title = {Stochastic Gradient {R}iemannian {L}angevin Dynamics on the Probability Simplex},
  Year = {2013}}

We use cython for the Gibbs sampling step which means you need to compile:

$ python setup.py build_ext --inplace

The entry point is run_wiki.py which runs the algorithm on articles from
wikipedia. Currently it's set to download articles as it is running which
is slow. It is recommended either downloading a batch and storing for
subsequent runs of the experiment, as detailed in the code, or using the
xml dumps of wikipedia here:

http://dumps.wikimedia.org/enwiki/latest/

We don't have code to process those files into the required form, but there
is some at 

http://www.ragtag.info/2011/feb/10/processing-every-wikipedia-article/

which could be adapted to do that.
