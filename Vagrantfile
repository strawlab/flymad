# encoding: utf-8

# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant::Config.run do |config|
  config.vm.box = "precise64"
  config.vm.box_url = "http://files.vagrantup.com/precise64.box"

  # If the following path does not exist, you need to do:
  #   cd packaging
  #   ./make-redist-sh.sh

  config.vm.provision :shell, :path => "packaging/install-flymad-on-ubuntu-12.04-amd64.sh"

end
