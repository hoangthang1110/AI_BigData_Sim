PGDMP                      }            ai_image_db    17.5    17.5     �           0    0    ENCODING    ENCODING        SET client_encoding = 'UTF8';
                           false            �           0    0 
   STDSTRINGS 
   STDSTRINGS     (   SET standard_conforming_strings = 'on';
                           false            �           0    0 
   SEARCHPATH 
   SEARCHPATH     8   SELECT pg_catalog.set_config('search_path', '', false);
                           false            �           1262    16454    ai_image_db    DATABASE     �   CREATE DATABASE ai_image_db WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'Vietnamese_Vietnam.1252';
    DROP DATABASE ai_image_db;
                     postgres    false            �            1259    16469    image_metadata    TABLE     �  CREATE TABLE public.image_metadata (
    id integer NOT NULL,
    file_name character varying(255) NOT NULL,
    image_path character varying(500) NOT NULL,
    category character varying(100),
    uploaded_by_user_id integer NOT NULL,
    upload_timestamp character varying(50) NOT NULL,
    ai_predicted_category character varying(255),
    ai_prediction_confidence double precision,
    ai_top_3_predictions text
);
 "   DROP TABLE public.image_metadata;
       public         heap r       postgres    false            �           0    0    TABLE image_metadata    ACL     �   REVOKE ALL ON TABLE public.image_metadata FROM postgres;
GRANT ALL ON TABLE public.image_metadata TO postgres WITH GRANT OPTION;
GRANT ALL ON TABLE public.image_metadata TO aiuser WITH GRANT OPTION;
          public               postgres    false    218            �            1259    16468    image_metadata_id_seq    SEQUENCE     �   CREATE SEQUENCE public.image_metadata_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 ,   DROP SEQUENCE public.image_metadata_id_seq;
       public               postgres    false    218            �           0    0    image_metadata_id_seq    SEQUENCE OWNED BY     O   ALTER SEQUENCE public.image_metadata_id_seq OWNED BY public.image_metadata.id;
          public               postgres    false    217            �            1259    16486    users    TABLE     �   CREATE TABLE public.users (
    id integer NOT NULL,
    username character varying(64) NOT NULL,
    password_hash character varying(256) NOT NULL
);
    DROP TABLE public.users;
       public         heap r       postgres    false            �           0    0    TABLE users    ACL     �   REVOKE ALL ON TABLE public.users FROM postgres;
GRANT ALL ON TABLE public.users TO postgres WITH GRANT OPTION;
GRANT ALL ON TABLE public.users TO aiuser;
          public               postgres    false    220            �            1259    16485    users_id_seq    SEQUENCE     �   CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 #   DROP SEQUENCE public.users_id_seq;
       public               postgres    false    220            �           0    0    users_id_seq    SEQUENCE OWNED BY     =   ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;
          public               postgres    false    219            (           2604    16472    image_metadata id    DEFAULT     v   ALTER TABLE ONLY public.image_metadata ALTER COLUMN id SET DEFAULT nextval('public.image_metadata_id_seq'::regclass);
 @   ALTER TABLE public.image_metadata ALTER COLUMN id DROP DEFAULT;
       public               postgres    false    218    217    218            )           2604    16489    users id    DEFAULT     d   ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);
 7   ALTER TABLE public.users ALTER COLUMN id DROP DEFAULT;
       public               postgres    false    219    220    220            �          0    16469    image_metadata 
   TABLE DATA           �   COPY public.image_metadata (id, file_name, image_path, category, uploaded_by_user_id, upload_timestamp, ai_predicted_category, ai_prediction_confidence, ai_top_3_predictions) FROM stdin;
    public               postgres    false    218   �       �          0    16486    users 
   TABLE DATA           <   COPY public.users (id, username, password_hash) FROM stdin;
    public               postgres    false    220   *       �           0    0    image_metadata_id_seq    SEQUENCE SET     D   SELECT pg_catalog.setval('public.image_metadata_id_seq', 15, true);
          public               postgres    false    217            �           0    0    users_id_seq    SEQUENCE SET     :   SELECT pg_catalog.setval('public.users_id_seq', 1, true);
          public               postgres    false    219            +           2606    16478 ,   image_metadata image_metadata_image_path_key 
   CONSTRAINT     m   ALTER TABLE ONLY public.image_metadata
    ADD CONSTRAINT image_metadata_image_path_key UNIQUE (image_path);
 V   ALTER TABLE ONLY public.image_metadata DROP CONSTRAINT image_metadata_image_path_key;
       public                 postgres    false    218            -           2606    16476 "   image_metadata image_metadata_pkey 
   CONSTRAINT     `   ALTER TABLE ONLY public.image_metadata
    ADD CONSTRAINT image_metadata_pkey PRIMARY KEY (id);
 L   ALTER TABLE ONLY public.image_metadata DROP CONSTRAINT image_metadata_pkey;
       public                 postgres    false    218            /           2606    16491    users users_pkey 
   CONSTRAINT     N   ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);
 :   ALTER TABLE ONLY public.users DROP CONSTRAINT users_pkey;
       public                 postgres    false    220            1           2606    16493    users users_username_key 
   CONSTRAINT     W   ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_username_key UNIQUE (username);
 B   ALTER TABLE ONLY public.users DROP CONSTRAINT users_username_key;
       public                 postgres    false    220                       826    16456    DEFAULT PRIVILEGES FOR TABLES    DEFAULT ACL     �   ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON TABLES TO postgres WITH GRANT OPTION;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public GRANT ALL ON TABLES TO aiuser;
          public               postgres    false                       826    16484    DEFAULT PRIVILEGES FOR TABLES    DEFAULT ACL     J   ALTER DEFAULT PRIVILEGES FOR ROLE postgres GRANT ALL ON TABLES TO aiuser;
                        postgres    false            �   k  x���͎�6 ��{l��W$EJ�[�{+���gc$ޙ�����/�l<�{���e��D���|�/Th��XB������ik��^������^��~1`���V>Z��3��e�������?cK����6R ��~J㫟?������T:YCԱ?���ևۘ�U���u��tXd��X�w�[s7���C$
L��hF:�*�}�z���^���5]�
���+�!�L�8���l0yb�{��th��^���~\�m������q;N(V��N5��8�ҥ����}��O�1U��]w�G��v3�bP�=l��	�Jұ��}�K���8�/�ۃ�
Mƨ;�j׭�1=2$�>7�om����Y2�ƒ�;�yJ^
�#�R��5펛&�r�n�ͺ����UaoY���GQ�{S��-���C��(U՗��k�yלvM����U��ܐ���Q�t]K���
b����;>7��t#@��!	���rŴu�i�����uSo_�}kWE�ns�׳����97��v������ C��ݔ����*n���2$�0"�ZZ
���'��=8��-Y�(<�ف'm�t�fw�ǧ	W�=:����K	Qk
ͮޟ�N����{(�*r�q�w���Cs=,S���n��ʅ��H�)�_Q�΋��I_�H &���Ddn���X��׾�O�K�[�R�Ϟ�v)pN�K=���MPk�b`�ڋ=5Ѐ�wTR*e������S�R�<U�g	nTҶ���A����.�+ߊ#}���V7&\�>�.�[i�ޕ��A�[�d���b�y���Rp��q}�T�8��q���-���(S#�#}�H�Y������mW�_�͵�pU<�EQ����      �   �   x��;�0 й9Gg�؉cwG�L,I�	����x/�\ޯ�ֶ�>ۄ��'��x���,���#�K��k�ЂgO�k�Q�z����4��UBBkՐ�J��ldУ����j�k+I$�ngM0i�ғ�/T�������1�     